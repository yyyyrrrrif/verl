# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import importlib.util
import json
import logging
import os
import time
from typing import Any, Callable, Protocol
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import ray
from cachetools import LRUCache

from verl.workers.config.rollout import RouterConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000

# ── Routing trace (long-tail experiment) ─────────────────────────────────
#
# Writes one JSONL record per acquire / release to a file so the long-tail
# behaviour of GlobalRequestLoadBalancer (long requests piling onto a few
# replicas while others sit idle, inflating total rollout time) can be analysed
# offline. Disabled unless ROUTER_TRACE=1 — the writer then short-circuits to a
# no-op, so there is zero overhead and no file I/O in production.

_DEFAULT_TRACE_PATH = "./router_trace_global_lb.jsonl"


class _RouterTraceWriter:
    """Append-only JSONL trace writer, env-gated.

    Records are one JSON object per line. Gated by ``ROUTER_TRACE``: when off,
    ``enabled`` is False and every ``append`` is a no-op fast path, so the
    load balancer pays nothing. When on, each ``append`` serialises the record
    and writes + flushes a single line (the LB actor runs acquire/release
    serially, so no locking is needed for the file handle).
    """

    _instance: "_RouterTraceWriter | None" = None

    def __init__(self) -> None:
        self.enabled = os.getenv("ROUTER_TRACE", "") == "1"
        self.path = os.getenv("ROUTER_TRACE_PATH", _DEFAULT_TRACE_PATH) if self.enabled else None

    @classmethod
    def get(cls) -> "_RouterTraceWriter":
        """Return the process-wide writer (lazily created)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def append(self, record: dict) -> None:
        """Append one trace record as a JSON line. No-op when disabled."""
        if not self.enabled or self.path is None:
            return
        line = json.dumps(record, separators=(",", ":")) + "\n"
        try:
            # Open per-append: avoids holding a file handle across the actor's
            # lifetime and is safe against concurrent actors writing the same
            # path (each line is a single write() syscall).
            with open(self.path, "a") as f:
                f.write(line)
        except OSError as exc:
            # Trace is a best-effort experiment harness — never break routing
            # on a write failure; one warning per failure is enough.
            logger.warning("[RouterTrace] write failed (%s); record dropped", exc)


_trace = _RouterTraceWriter.get()

# ── Conditional migration knobs (long-tail mitigation) ────────────────────
#
# When a sticky-bound replica is a *straggler* AND idle replicas exist to
# absorb the load, the balancer migrates the request's *future* turns to the
# lightest replica (paying one prefix-cache recompute to shed the long-tail).
#
# The judgment is RELATIVE, not an absolute inflight threshold. Trace analysis
# (A-group sticky) showed the tail forms at inflight 6-9 on the straggler while
# the rest have drained to 0 — an absolute overload threshold (e.g. 12) can
# never fire there because the tail's absolute level sits below the steady-state
# peak, so it only triggered during warm-up (8/6818 acquires, none in the tail).
# The features that actually separate "tail" from "steady overload":
#   * bound inflight relative to the fleet mean  (straggler, ranked at front)
#   * number of near-idle replicas               (headroom exists → it's a tail)
# Raw inflight variance does NOT separate them (steady≈tail); the idle count /
# normalised variance (CV) does.
#
# Default OFF: OVER_FACTOR defaults to +inf so no migration ever happens unless
# the user sets ROUTER_MIGRATE_OVER_FACTOR. Calibrated starting point from the
# A-group trace: OVER_FACTOR=1.5, MIN_IDLE=2, IDLE_THRESH=1, GAP=4, MIN_ABS=3
# (fires across mid+tail, 0× in warm-up).
_MIGRATE_OVER_FACTOR = float(os.getenv("ROUTER_MIGRATE_OVER_FACTOR", "inf"))
_MIGRATE_MIN_IDLE = int(os.getenv("ROUTER_MIGRATE_MIN_IDLE", "2"))
_MIGRATE_IDLE_THRESH = int(os.getenv("ROUTER_MIGRATE_IDLE_THRESH", "1"))
_MIGRATE_GAP = float(os.getenv("ROUTER_MIGRATE_GAP", "4"))
_MIGRATE_MIN_ABS = int(os.getenv("ROUTER_MIGRATE_MIN_ABS", "3"))


class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers.

    All strategies must satisfy this interface via structural subtyping.
    """

    def acquire_server(
        self, request_id: str, prompt_ids: list[int] | None = None
    ) -> tuple[str, Any]:
        """Acquire a server for the given request.

        Args:
            request_id: Request identifier for sticky session routing.
            prompt_ids: Prompt token ids for content-aware routing.

        Returns:
            A ``(server_id, actor_handle)`` tuple.

        Raises:
            RuntimeError: If no servers are available in the pool.
        """
        ...

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        Args:
            server_id: Identifier of the server to release.
            request_id: Optional request id of the completing turn. When the
                load balancer supports per-turn dwell-time tracing it pairs this
                with the matching ``acquire_server`` to record how long the
                turn occupied the replica; absent (``None``) it just decrements
                the in-flight counter. Optional so existing callers keep working.
        """
        ...

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the load balancer pool.

        Args:
            servers: Mapping from ``server_id`` to ``actor_handle``.
        """
        ...

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the load balancer pool.

        Args:
            server_ids: List of server identifiers to remove.
        """
        ...

    def get_all_servers(self) -> list[str]:
        """List all active server IDs.

        Returns:
            List of server identifier strings.
        """
        ...

    def get_status(self) -> dict:
        """Return current load balancer state for debugging.

        Returns:
            A dictionary with ``servers``, ``total_inflight``,
            and ``active_servers`` keys.
        """
        ...


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(self, servers: dict[str, ray.actor.ActorHandle], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("servers must be non-empty")

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        # Long-tail trace state (see plan-19). Only touched when tracing is on.
        # Per-request acquire count (the "turn" — LB's view of which turn of the
        # conversation this acquire serves, since agent-layer turn counts never
        # reach the LB).
        self._request_turn_count: dict[str, int] = {}
        # request_id → stack of (turn, acquire_ts) for in-flight acquires.
        # Stack because a request_id can be re-acquired before a prior release
        # lands (rare, but possible under async); release pops the matching one.
        self._pending: dict[str, list[tuple[int, float]]] = {}

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, ray.actor.ActorHandle]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        is_sticky = False
        migrated = False
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                # Conditional migration (long-tail mitigation): if the bound
                # replica is a straggler relative to the fleet AND near-idle
                # replicas exist to absorb it, migrate this request's future
                # turns to the lightest replica — paying one prefix-cache
                # recompute to shed the long-tail. Relative + headroom-gated by
                # construction: in steady state every replica is busy so the
                # idle-count condition fails and sticky is preserved.
                target = self._pick_migration_target(server_id)
                if target is not None:
                    # Migrate: rebind sticky to the lightest replica so the
                    # following turns stay there (preserve its prefix cache).
                    self._request_id_to_server[request_id] = target
                    self._inflight_requests[target] += 1
                    migrated = True
                    return (
                        self._trace_acquire(request_id, target, is_sticky=False, migrated=True),
                        self._servers[target],
                    )
                # Not migrated: keep sticky on the bound replica.
                self._inflight_requests[server_id] += 1
                is_sticky = True
                return self._trace_acquire(request_id, server_id, is_sticky, migrated=False), self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return self._trace_acquire(request_id, server_id, is_sticky, migrated=False), self._servers[server_id]

    def _pick_migration_target(self, bound_server_id: str) -> str | None:
        """Return a lighter replica to migrate to, or ``None`` to stay sticky.

        Long-tail is a *relative* phenomenon: in the tail a straggler replica
        sits at inflight 6-9 while the rest have drained to 0. An absolute
        overload threshold can't see this (the tail's absolute level is well
        below the steady-state peak), so the judgment is relative to the current
        fleet. Migration fires only when ALL hold:

          (A) the bound replica is a straggler — its inflight is at least
              ``_MIGRATE_OVER_FACTOR`` times the fleet mean (ranked at the
              front);
          (B) real headroom exists — at least ``_MIGRATE_MIN_IDLE`` replicas are
              near-idle (inflight <= ``_MIGRATE_IDLE_THRESH``). This is what
              separates the tail (some replicas drained) from steady-state
              overload (all replicas busy), where migrating would only thrash
              prefix cache;
          (C) the lightest replica is at least ``_MIGRATE_GAP`` lighter than the
              bound one;
          (D) floor — bound inflight is at least ``_MIGRATE_MIN_ABS``, so the
              final two or three requests aren't churned (migration costs a
              prefix recompute a near-drained fleet can't repay).

        When all hold, returns the globally lightest other replica (so the
        migrated turn lands where there's most headroom). Otherwise ``None`` —
        the caller keeps the sticky binding.

        Note: ``bound_inflight`` here is the count *before* this turn's acquire
        is added (the increment happens in the caller), i.e. the replica's
        currently-running turns.

        Ported to the KVCAwareBalancer, (A)/(B)/(C) swap the inflight count for a
        ``kv_usage + running + waiting`` load metric; the structure is unchanged.
        """
        if _MIGRATE_OVER_FACTOR == float("inf"):
            return None  # migration disabled (default)
        counts = self._inflight_requests
        if len(counts) < 2:
            return None  # nowhere to migrate to
        bound_inflight = counts.get(bound_server_id, 0)
        if bound_inflight < _MIGRATE_MIN_ABS:
            return None  # (D) floor: too few inflight to be worth churning
        mean = sum(counts.values()) / len(counts)
        if bound_inflight < _MIGRATE_OVER_FACTOR * mean:
            return None  # (A) bound is not a straggler relative to the fleet
        n_idle = sum(1 for c in counts.values() if c <= _MIGRATE_IDLE_THRESH)
        if n_idle < _MIGRATE_MIN_IDLE:
            return None  # (B) no headroom → steady overload, not a tail
        candidates = {sid: c for sid, c in counts.items() if sid != bound_server_id}
        lightest_id = min(candidates, key=candidates.get)
        if bound_inflight - candidates[lightest_id] < _MIGRATE_GAP:
            return None  # (C) no sufficiently lighter replica
        return lightest_id

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        When ``request_id`` is supplied it is paired with the matching acquire
        to emit a dwell-time trace record (how long the turn occupied the
        replica); otherwise only the in-flight counter is decremented.
        """
        inflight_after: int | None = None
        if server_id in self._inflight_requests:
            if self._inflight_requests[server_id] > 0:
                self._inflight_requests[server_id] -= 1
            inflight_after = self._inflight_requests[server_id]
        self._trace_release(server_id, request_id, inflight_after)

    # ── Trace helpers (long-tail experiment) ────────────────────────────

    def _trace_acquire(self, request_id: str, server_id: str, is_sticky: bool, migrated: bool = False) -> str:
        """Record the turn, push (turn, ts) onto the request's pending stack.

        Always returns ``server_id`` so callers can inline it into their return.
        No-ops to a fast path when tracing is disabled.
        """
        turn = self._request_turn_count.get(request_id, 0) + 1
        self._request_turn_count[request_id] = turn
        now = time.time()
        if _trace.enabled:
            self._pending.setdefault(request_id, []).append((turn, now))
            _trace.append(
                {
                    "event": "acquire",
                    "ts": now,
                    "request_id": request_id,
                    "turn": turn,
                    "replica_id": server_id,
                    "is_sticky": is_sticky,
                    "migrated": migrated,
                    "inflight_after": self._inflight_requests.get(server_id, 0),
                }
            )
        return server_id

    def _trace_release(self, server_id: str, request_id: str | None, inflight_after: int | None) -> None:
        """Pop the matching acquire and emit a dwell-time release record."""
        if not _trace.enabled or request_id is None:
            return
        stack = self._pending.get(request_id)
        if not stack:
            # Acquire wasn't traced (e.g. it predates tracing-on) — nothing to pair.
            return
        turn, acquire_ts = stack.pop()
        now = time.time()
        _trace.append(
            {
                "event": "release",
                "ts": now,
                "request_id": request_id,
                "turn": turn,
                "replica_id": server_id,
                "dwell_sec": round(now - acquire_ts, 6),
                "inflight_after": inflight_after if inflight_after is not None else 0,
            }
        )

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }


class LoadBalancerRegistry:
    """Registry for load-balancer strategy factory functions.

    Strategies are registered by name and looked up via :meth:`get`.
    The ``plugin_extension`` strategy dynamically imports
    a user-defined class from ``router.router_class``.
    """

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., Any]) -> None:
        """Register a load-balancer strategy factory function."""
        if name in cls._registry:
            raise ValueError(
                f"Load balancer '{name}' is already registered. "
                f"Existing factory: {cls._registry[name]}"
            )
        cls._registry[name] = factory
        logger.info("Registered load balancer strategy: %s", name)

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        """Look up a registered factory by name."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown load balancer strategy: '{name}'. "
                f"Available strategies: {cls.list_strategies()}"
            )
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._registry.keys())


def _create_global_sticky_inflight(
    servers: dict[str, Any],
    router_config: RouterConfig | None = None,
):
    """Factory for the default sticky-session + least-inflight strategy.
    """

    return GlobalRequestLoadBalancer.remote(
        servers=servers,
        max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
    )

def _resolve_config_path(config_path: str) -> str:
    """Resolve a router config path to an absolute filesystem path.

    Supports two forms:

    - ``pkg://<package>/<rel/path>``: resolved against an installed Python
      package directory (works for regular packages and namespace dirs).
      Example: ``pkg://uni_agent.llm_router.configs/kvc_aware_router.yaml``.
    - Any other value: treated as a filesystem path (absolute or CWD-relative).

    Returns the absolute path; raises ``ValueError``/``ImportError`` on bad input.
    """
    if config_path.startswith("pkg://"):
        rest = config_path[len("pkg://"):]
        pkg_name, sep, rel_path = rest.partition("/")
        if not sep or not rel_path:
            raise ValueError(
                f"Invalid pkg:// URI '{config_path}': expected "
                f"'pkg://<package>/<relative/path>'"
            )
        try:
            spec = importlib.util.find_spec(pkg_name)
        except (ImportError, ValueError) as e:
            raise ImportError(f"Cannot resolve package '{pkg_name}': {e}") from e
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(
                f"Package '{pkg_name}' not found or has no __path__ "
                f"(is it installed?)."
            )
        pkg_dir = os.path.abspath(next(iter(spec.submodule_search_locations)))
        return os.path.join(pkg_dir, rel_path)
    return os.path.abspath(config_path)


def _load_router_yaml(router_config: RouterConfig) -> dict:
    """Load a router YAML configuration, resolving Hydra ``defaults`` composition.

    Unlike ``OmegaConf.load``, this expands the ``defaults`` block so referenced
    sub-configs (strategies, collectors, cache_store) are merged into the final
    config. ``router_config_path`` is resolved via :func:`_resolve_config_path`
    (supports ``pkg://`` package-relative URIs and plain filesystem paths).
    """
    config_path = router_config.get("router_config_path", None)
    if not config_path:
        raise ValueError(
            "The 'plugin_extension' strategy requires 'router_config_path' "
            "pointing to a YAML file."
        )

    full_path = _resolve_config_path(config_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Router config file not found: {full_path}")

    config_dir = os.path.dirname(full_path)
    config_name = os.path.basename(full_path)
    for ext in (".yaml", ".yml"):
        if config_name.endswith(ext):
            config_name = config_name[: -len(ext)]
            break

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg: DictConfig = compose(config_name=config_name)
    return OmegaConf.to_container(cfg, resolve=True)

def _resolve_router_class(yaml_config: dict) -> type:
    """Validate and import a router class from YAML config.
    Extracts ``router_class`` FQN, validates format, imports the module,
    and returns the class object.
    """
    router_class = yaml_config.get("router_class", None)
    if not router_class:
        raise ValueError(
            "External router YAML must contain 'router_class'. "
            "Example: router_class: uni_agent.llm_router.KvcAwareRouter"
        )

    try:
        module_path, class_name = router_class.rsplit(".", 1)
    except ValueError:
        raise ValueError(
            f"Invalid fully-qualified class name: '{router_class}'. "
            f"Expected format: 'module_path.ClassName'"
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' for '{class_name}'. "
            f"Original error: {e}"
        ) from e

    cls = getattr(module, class_name)  # AttributeError propagates if missing

    if not callable(cls):
        raise TypeError(
            f"'{router_class}' is not callable (type: {type(cls).__name__}). "
            f"Expected a class with a .remote() constructor."
        )

    return cls


def _create_plugin_extension(
    servers: dict[str, Any],
    router_config: RouterConfig | None = None,
):
    """Factory for user-defined load balancer via external YAML configuration.

    Loads the class specified by ``router_config_path``, imports it
    dynamically, and instantiates it as a Ray actor with ``servers``
    and YAML kwargs.
    """

    yaml_config = _load_router_yaml(router_config)
    cls = _resolve_router_class(yaml_config)

    ray_cls = cls if isinstance(cls, ray.actor.ActorClass) else ray.remote(cls)

    logger.info(
        "Creating plugin load balancer: class=%s, servers=%d, kwargs=%s",
        yaml_config["router_class"],
        len(servers),
        yaml_config,
    )
    return ray_cls.remote(servers, yaml_config)


LoadBalancerRegistry.register("global_sticky_inflight", _create_global_sticky_inflight)
LoadBalancerRegistry.register("plugin_extension", _create_plugin_extension)


def get_router_handle(servers: dict[str, Any], router_config: RouterConfig = None) -> Any:
    """Create a load balancer instance from router configuration."""
    if router_config is None:
        strategy = "global_sticky_inflight"
    else:
        strategy = router_config.get("router_strategy", "global_sticky_inflight")

    factory = LoadBalancerRegistry.get(strategy)
    return factory(servers=servers, router_config=router_config)
