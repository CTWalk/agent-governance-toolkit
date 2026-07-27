# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Expose ACS to any agent-hooks host as an :class:`Interceptor`.

The `agent-hooks <https://github.com/responsibleai/agent-hooks>`_ contract is
the framework-neutral seam that agent runtimes (crewAI, LangGraph, AutoGen, ...)
already speak: a host emits a wire-shaped ``AgentContext`` at each lifecycle
interception point and enforces the ``Verdict`` an *interceptor* returns.

:class:`AcsInterceptor` turns an ACS :class:`~agt.policies.runtime.AgtRuntime`
into exactly such an interceptor. A single instance therefore lets *every*
agent-hooks host inject ACS governance with no framework-specific glue::

    from agt.policies.agent_hooks import AcsInterceptor

    acs = AcsInterceptor.from_manifest("governance.yaml")

    # crewAI: inject ACS into every agent in the crew.
    from crewai.hooks import use_agent_hooks

    with use_agent_hooks(acs):
        crew.kickoff(inputs={"topic": "quarterly report"})

For each emitted context the interceptor

1. reads the interception point (``context["interception_point"]``),
2. translates the agent-hooks context into the AGT snapshot shape
   (:class:`~agt.policies.snapshot.SnapshotBuilder`),
3. evaluates it through the runtime, and
4. maps the returned :class:`~agt.policies.result.PolicyEvaluation` back to an
   agent-hooks verdict.

The interceptor returns a *wire-shaped* verdict ``dict`` rather than importing
``agent_hooks`` types, so this module stays importable in hosts that only speak
the wire format; the host normalizes and validates it (agent-hooks §5). The
five ACS verdicts are mapped onto the three agent-hooks decisions: ``allow``
and ``deny`` pass through, ``warn`` becomes an ``allow`` carrying a warning,
``escalate`` becomes a liftable ``deny`` (a ``deny`` with an ``approval``
block), and ``transform`` carries the materialized replacement.

Fail-closed: a missing interception point, an untranslatable/oversized context,
a snapshot-build error, or a runtime error is turned into a ``deny``. ACS is a
fail-closed decision runtime; this adapter never turns an error into an
``allow``.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, cast

from agt.policies.result import PolicyEvaluation
from agt.policies.snapshot import SnapshotBuilder

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from agt.policies.manifest import AgtManifest
    from agt.policies.runtime import ApprovalCallback


logger = logging.getLogger(__name__)


#: Wire-shaped agent-hooks context (agent-hooks §4): a JSON object.
AgentContext = dict[str, Any]
#: Wire-shaped agent-hooks verdict (agent-hooks §5); the host validates it.
WireVerdict = dict[str, Any]


# The eight ACS and agent-hooks intervention points share identical string
# names, so no name translation is needed between the two contracts.
_KNOWN_POINTS: frozenset[str] = frozenset(
    {
        "agent_startup",
        "input",
        "pre_model_call",
        "post_model_call",
        "pre_tool_call",
        "post_tool_call",
        "output",
        "agent_shutdown",
    }
)

# Points whose agent-hooks ``target`` is a ``{"content": ...}`` envelope, so a
# transform must be rooted at ``$target.content`` rather than ``$target``.
_CONTENT_ENVELOPE_POINTS: frozenset[str] = frozenset(
    {"input", "output", "post_model_call"}
)

_GENERIC_DENY_MESSAGE = "Request blocked by Agent Control Specification."

# ACS fails closed on an intervention point a manifest does not declare, tagging
# the result with this reason code. agent-hooks hosts emit all eight lifecycle
# points regardless of the manifest, so the adapter treats this specific signal
# as "not governed here" and passes the action through.
_UNGOVERNED_POINT_REASON = "runtime_error:intervention_point_unknown"

# Upper bound on the per-``(session, agent)`` snapshot-builder cache. The keys
# derive from untrusted context ids, so an unbounded cache is a memory-DoS
# vector for a long-lived, high-cardinality (e.g. multi-tenant) host. The cache
# is a plain per-session budget carrier, so evicting the least-recently-used
# entry only resets that session's budget counters — it never affects a policy
# decision, which ACS makes statelessly per snapshot.
_MAX_BUILDERS = 4096

# Adapter-synthesized fail-closed reasons. They are namespaced ``acs_adapter:``
# and deliberately avoid the agent-hooks-reserved ``host_error:`` prefix (§11)
# and the ACS-reserved ``runtime_error:`` prefix.
_REASON_CONTEXT_INVALID = "acs_adapter:context_invalid"
_REASON_SNAPSHOT_ERROR = "acs_adapter:snapshot_error"
_REASON_RUNTIME_ERROR = "acs_adapter:runtime_error"
_REASON_TRANSFORM_UNAVAILABLE = "acs_adapter:transform_unavailable"


class _Evaluator(Protocol):
    """Minimal structural view of :class:`~agt.policies.runtime.AgtRuntime`.

    Depending on this protocol (rather than the concrete runtime) keeps the
    adapter importable and unit-testable without the native ACS core, and lets
    hosts inject their own evaluator.
    """

    def evaluate(
        self, ip: str, snapshot: Mapping[str, Any], mode: str = ...
    ) -> PolicyEvaluation: ...


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce an untrusted context member into a ``dict[str, Any]``."""
    if isinstance(value, dict):
        typed = cast("dict[Any, Any]", value)
        return {str(key): item for key, item in typed.items()}
    return {}


def _as_list(value: Any) -> list[Any]:
    """Coerce an untrusted context member into a ``list``."""
    if isinstance(value, (list, tuple)):
        return list(cast("list[Any] | tuple[Any, ...]", value))
    return []


def _as_text(value: Any) -> str:
    """Render an untrusted scalar/object as text for a snapshot body."""
    if isinstance(value, str):
        return value
    return str(value)


def _as_body(value: Any) -> str | dict[str, Any]:
    """Coerce an untrusted content member into a snapshot body."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _as_mapping(value)
    return _as_text(value)


def _int_or_zero(value: Any) -> int:
    """Return ``value`` if it is a non-bool ``int``, else ``0``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _deny(reason: str, message: str) -> WireVerdict:
    """Build a fail-closed ``deny`` wire verdict."""
    return {"decision": "deny", "reason": reason, "message": message}


def _derive_governed_points(runtime: object) -> frozenset[str] | None:
    """Best-effort set of intervention points a runtime's manifest declares.

    Returns ``None`` when the set cannot be determined (e.g. an injected
    evaluator that exposes no manifest), in which case the adapter evaluates
    every point and relies on the ``intervention_point_unknown`` fallback.
    """
    manifest = getattr(runtime, "manifest", None)
    points = getattr(manifest, "intervention_points", None)
    if points is None:
        return None
    try:
        return frozenset(str(point) for point in points)
    except TypeError:
        return None


class AcsInterceptor:
    """An agent-hooks ``Interceptor`` backed by an ACS ``AgtRuntime``.

    Construct one with :meth:`from_manifest` (the common path) or
    :meth:`from_runtime`, then hand it to any agent-hooks host. The interceptor
    is stateless with respect to policy decisions — ACS evaluates each snapshot
    independently — but it keeps one :class:`SnapshotBuilder` per
    ``(session, agent)`` pair so per-session budget counters (tool calls,
    tokens) accumulate across a run.

    Instances are safe to share across the threads/tasks of a single host: the
    per-session builder cache and its budget mutations are guarded by a lock.
    """

    __slots__ = ("_builders", "_governed_points", "_lock", "_mode", "_runtime")

    def __init__(
        self,
        runtime: _Evaluator,
        *,
        mode: str = "enforce",
        governed_points: frozenset[str] | None = None,
    ) -> None:
        """Wrap ``runtime`` as an interceptor.

        Args:
            runtime: Any object exposing ``evaluate(ip, snapshot, mode)`` —
                normally an :class:`~agt.policies.runtime.AgtRuntime`.
            mode: ACS enforcement mode, ``"enforce"`` (default) or
                ``"evaluate_only"``. In ``evaluate_only`` the runtime records
                decisions without applying transforms or resolving approvals.
            governed_points: The intervention points this interceptor should
                evaluate. Points outside the set pass through as ``allow``,
                because an agent-hooks host emits every lifecycle point but a
                manifest governs only a subset. Defaults to the points the
                runtime's manifest declares (derived automatically).
        """
        self._runtime = runtime
        self._mode = mode
        self._builders: OrderedDict[str, SnapshotBuilder] = OrderedDict()
        self._lock = threading.Lock()
        if governed_points is None:
            governed_points = _derive_governed_points(runtime)
        self._governed_points = governed_points

    @classmethod
    def from_manifest(
        cls,
        manifest: Path | str | Mapping[str, Any] | AgtManifest,
        *,
        base_dir: Path | str | None = None,
        mode: str = "enforce",
        approval_resolver: ApprovalCallback | None = None,
        policy_dispatcher: Any | None = None,
        annotator_dispatcher: Any | None = None,
    ) -> AcsInterceptor:
        """Build an interceptor from an AGT manifest.

        This imports :class:`~agt.policies.runtime.AgtRuntime` lazily, so the
        module stays importable without the native ACS core installed.

        Args:
            manifest: A manifest path, YAML text, mapping, or typed
                :class:`AgtManifest`. Relative references in non-path inputs
                require ``base_dir``.
            base_dir: Provenance root for resolving relative manifest
                references when ``manifest`` is not a path.
            mode: ACS enforcement mode (see :meth:`__init__`).
            approval_resolver: Host approval callback invoked for ``escalate``
                verdicts (AGT-DELTA D1.4). When ``None`` an ``escalate`` is
                surfaced as a liftable ``deny``.
            policy_dispatcher: Optional host policy dispatcher (ACS §12.3) — for
                ``custom`` policies or to run without the bundled OPA dispatcher.
            annotator_dispatcher: Optional host annotator dispatcher.

        Returns:
            A ready-to-register :class:`AcsInterceptor`.
        """
        from agt.policies.runtime import AgtRuntime

        runtime = AgtRuntime.from_manifest(
            manifest,
            base_dir=base_dir,
            approval_resolver=approval_resolver,
            policy_dispatcher=policy_dispatcher,
            annotator_dispatcher=annotator_dispatcher,
        )
        return cls(runtime, mode=mode)

    @classmethod
    def from_runtime(
        cls, runtime: _Evaluator, *, mode: str = "enforce"
    ) -> AcsInterceptor:
        """Build an interceptor from an already-constructed runtime."""
        return cls(runtime, mode=mode)

    # -- agent-hooks Interceptor contract -------------------------------------

    def intercept(self, context: AgentContext, /) -> WireVerdict:
        """Evaluate one agent-hooks context and return a wire verdict.

        This is the agent-hooks ``Interceptor`` entry point. It never raises:
        every failure is mapped to a fail-closed ``deny`` so the host blocks
        rather than proceeding ungoverned.
        """
        point = context.get("interception_point")
        if not isinstance(point, str) or point not in _KNOWN_POINTS:
            logger.warning("ACS interceptor: unknown interception point %r", point)
            return _deny(_REASON_CONTEXT_INVALID, _GENERIC_DENY_MESSAGE)

        if self._governed_points is not None and point not in self._governed_points:
            # The manifest does not govern this point; pass the action through.
            return {"decision": "allow"}

        try:
            snapshot = self._build_snapshot(point, context)
        except Exception:
            logger.exception("ACS interceptor: snapshot build failed at %s", point)
            return _deny(_REASON_SNAPSHOT_ERROR, _GENERIC_DENY_MESSAGE)

        try:
            evaluation = self._runtime.evaluate(point, snapshot, self._mode)
        except Exception:
            logger.exception("ACS interceptor: evaluation failed at %s", point)
            return _deny(_REASON_RUNTIME_ERROR, _GENERIC_DENY_MESSAGE)

        if (
            self._governed_points is None
            and evaluation.verdict == "deny"
            and evaluation.reason_code == _UNGOVERNED_POINT_REASON
        ):
            # Defense in depth ONLY when the governed set could not be derived
            # (an injected evaluator exposing no manifest): an undeclared point
            # is "not governed", not denied. When the governed set IS known, a
            # governed point's deny stays a deny — a policy that fails to bind
            # must fail closed, never open.
            return {"decision": "allow"}

        self._record_budget(point, context)
        verdict = _to_wire_verdict(evaluation, point)
        logger.debug(
            "ACS interceptor: %s -> %s (%s)",
            point,
            evaluation.verdict,
            evaluation.reason_code or "-",
        )
        return verdict

    # -- snapshot translation -------------------------------------------------

    def _builder(self, context: AgentContext) -> SnapshotBuilder:
        """Return the :class:`SnapshotBuilder` for this context's session."""
        agent = _as_mapping(context.get("agent"))
        session = _as_mapping(context.get("session"))
        agent_id = _as_text(agent.get("id") or "agent")
        session_id = _as_text(session.get("id") or "session")
        name = agent.get("name")
        agent_name = name if isinstance(name, str) else None
        tenant = _as_mapping(context.get("tenant"))
        tenant_id = tenant.get("id") if isinstance(tenant.get("id"), str) else None
        key = f"{session_id}\x00{agent_id}"
        with self._lock:
            builder = self._builders.get(key)
            if builder is None:
                builder = SnapshotBuilder(
                    agent_id=agent_id,
                    session_id=session_id,
                    agent_name=agent_name,
                    tenant_id=tenant_id,
                )
                self._builders[key] = builder
                # Bound the cache: evict the least-recently-used session when
                # it grows past the cap (see _MAX_BUILDERS).
                if len(self._builders) > _MAX_BUILDERS:
                    self._builders.popitem(last=False)
            else:
                self._builders.move_to_end(key)
            return builder

    def _build_snapshot(self, point: str, context: AgentContext) -> dict[str, Any]:
        """Translate an agent-hooks context into an AGT snapshot for ``point``."""
        builder = self._builder(context)

        if point == "input":
            inp = _as_mapping(context.get("input"))
            role = inp.get("role")
            return builder.input(
                body=_as_body(inp.get("content")),
                source=role if isinstance(role, str) else "user",
            )

        if point == "pre_model_call":
            model = _as_mapping(context.get("model"))
            return builder.pre_model_call(
                model_name=_as_text(model.get("id") or "unknown"),
                messages=_as_list(context.get("messages")),
            )

        if point == "post_model_call":
            model = _as_mapping(context.get("model"))
            usage_raw = context.get("usage")
            return builder.post_model_call(
                model_name=_as_text(model.get("id") or "unknown"),
                response=_as_mapping(context.get("response")),
                usage=_as_mapping(usage_raw) if isinstance(usage_raw, dict) else None,
            )

        if point == "pre_tool_call":
            tool_call = _as_mapping(context.get("tool_call"))
            return builder.pre_tool_call(
                tool_name=_as_text(tool_call.get("name") or ""),
                args=_as_mapping(tool_call.get("args")),
                call_id=_as_text(tool_call.get("id") or "call-1"),
            )

        if point == "post_tool_call":
            tool_call = _as_mapping(context.get("tool_call"))
            tool_result = _as_mapping(context.get("tool_result"))
            duration = tool_result.get("duration_ms")
            return builder.post_tool_call(
                tool_name=_as_text(tool_call.get("name") or ""),
                args=_as_mapping(tool_call.get("args")),
                result=tool_result.get("value"),
                error="error" if tool_result.get("is_error") else None,
                duration_ms=float(duration)
                if isinstance(duration, (int, float))
                else 0.0,
                call_id=_as_text(tool_call.get("id") or "call-1"),
            )

        if point == "output":
            out = _as_mapping(context.get("output"))
            return builder.output(content=_as_body(out.get("content")))

        if point == "agent_startup":
            init = _as_mapping(context.get("agent_init"))
            return builder.agent_startup(
                tools_registered=[
                    _as_text(t) for t in _as_list(init.get("tools_registered"))
                ],
                capabilities=[_as_text(c) for c in _as_list(init.get("capabilities"))],
            )

        # point == "agent_shutdown" (the only remaining known point).
        return builder.agent_shutdown()

    def _record_budget(self, point: str, context: AgentContext) -> None:
        """Advance host-side budget counters after a completed action.

        Counters are read at the start of each ACS evaluation, so recording a
        completed tool call / token spend here surfaces it to the *next*
        intervention point. Best-effort and never raises.
        """
        if point == "post_tool_call":
            builder = self._builder(context)
            with self._lock:
                builder.record_tool_call()
        elif point == "post_model_call":
            usage = _as_mapping(context.get("usage"))
            total = _int_or_zero(usage.get("total_tokens"))
            if total == 0:
                total = _int_or_zero(usage.get("prompt_tokens")) + _int_or_zero(
                    usage.get("completion_tokens")
                )
            if total > 0:
                builder = self._builder(context)
                with self._lock:
                    builder.record_tokens(total)


# -- verdict mapping ----------------------------------------------------------


def _transform_root(point: str) -> str:
    """Return the ``$target`` root a transform must use for ``point``."""
    if point in _CONTENT_ENVELOPE_POINTS:
        return "$target.content"
    return "$target"


def _to_wire_verdict(evaluation: PolicyEvaluation, point: str) -> WireVerdict:
    """Map an ACS :class:`PolicyEvaluation` to an agent-hooks wire verdict.

    ACS exposes five verdicts; agent-hooks encodes three decisions (§5.1):

    - ``allow``    -> ``allow``
    - ``warn``     -> ``allow`` + a ``warnings`` entry
    - ``transform``-> ``transform`` with a ``$target``-rooted replacement
    - ``deny``     -> ``deny`` (reason + full message)
    - ``escalate`` -> ``deny`` + an ``approval`` block (a liftable deny, §9)
    """
    verdict = evaluation.verdict
    reason = evaluation.reason_code or None
    message = evaluation.message or None

    if verdict == "transform":
        return _transform_verdict(evaluation, point)

    if verdict in ("deny", "escalate"):
        out: WireVerdict = {
            "decision": "deny",
            "reason": reason or "policy:blocked",
            "message": message or evaluation.public_error_message(),
        }
        if verdict == "escalate":
            # A liftable deny: the host approval seam (agent-hooks §9) MAY lift
            # it; without a resolver it stays denied and fails closed.
            out["approval"] = {}
        return out

    # allow / warn both permit the action in agent-hooks terms.
    allow: WireVerdict = {"decision": "allow"}
    if verdict == "warn":
        warning: dict[str, Any] = {}
        if reason is not None:
            warning["reason"] = reason
        if message is not None:
            warning["message"] = message
        allow["warnings"] = [warning]
    labels = list(evaluation.result_labels)
    if labels:
        allow["result_labels"] = labels
    _attach_evidence(allow, evaluation)
    return allow


def _transform_verdict(evaluation: PolicyEvaluation, point: str) -> WireVerdict:
    """Map an ACS ``transform`` verdict onto an agent-hooks transform.

    ACS confines a transform to the policy target and returns the materialized
    replacement in ``transform.applied_value`` (falling back to the declared
    ``value``). Because the agent-hooks ``target`` mirrors the policy target,
    the replacement is re-rooted at ``$target`` (or ``$target.content`` for the
    content-envelope points). If no replacement value is available the verdict
    fails closed to ``deny`` rather than proceeding unmodified.
    """
    transform = evaluation.transform
    if transform is None:
        return _deny(_REASON_TRANSFORM_UNAVAILABLE, _GENERIC_DENY_MESSAGE)
    value = (
        transform.applied_value
        if transform.applied_value is not None
        else transform.value
    )
    if value is None:
        return _deny(_REASON_TRANSFORM_UNAVAILABLE, _GENERIC_DENY_MESSAGE)
    out: WireVerdict = {
        "decision": "transform",
        "transform": {"path": _transform_root(point), "value": value},
    }
    if evaluation.reason_code:
        out["reason"] = evaluation.reason_code
    if evaluation.message:
        out["message"] = evaluation.message
    labels = list(evaluation.result_labels)
    if labels:
        out["result_labels"] = labels
    _attach_evidence(out, evaluation)
    return out


def _attach_evidence(verdict: WireVerdict, evaluation: PolicyEvaluation) -> None:
    """Copy any ACS evidence onto ``verdict`` in the agent-hooks wire shape."""
    evidence = evaluation.evidence
    if evidence is None:
        return
    wire: dict[str, Any] = {}
    if evidence.artefact is not None:
        wire["artefact"] = _as_text(evidence.artefact)
    pointers = {str(k): _as_text(v) for k, v in evidence.verification_pointers.items()}
    if pointers:
        wire["verification_pointers"] = pointers
    if wire:
        verdict["evidence"] = wire


__all__ = ["AcsInterceptor", "AgentContext", "WireVerdict"]
