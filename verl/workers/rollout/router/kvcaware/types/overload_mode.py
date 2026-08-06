# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sticky-overload check mode.

``is_overloaded`` decides whether the sticky short-circuit sends a returning
session back to its bound replica (``memory_overload_filter=True``) or falls
back to combined scoring. The mode is independent of ``slow_cut`` so the
overload signal can be chosen without coupling to the routing strategy.

Canonical YAML strings (``simple`` / ``blended``) are mapped to these
constants at the config boundary (``KVCAwareStrategyConfig``).
"""

from __future__ import annotations

from enum import Enum


class OverloadMode(str, Enum):
    """How ``is_overloaded`` decides a replica is overloaded."""

    # kv_cache_usage_perc > load_threshold — a single vLLM occupancy signal
    KV_CACHE_USAGE_PERC = "kv_cache_usage_perc"
    # _compute_load(kv, running, waiting, inflight) > load_threshold
    KV_LOAD = "kv_load"
    # num_requests_waiting > 0 — vLLM is queueing requests on this replica, a
    # direct result-level evidence of KV-capacity overload (the request did not
    # fit in the running set and was deferred to the waiting queue). Unlike the
    # occupancy signals above, this is binary (no load_threshold) and its
    # trigger rate adapts to the context: loose capacity (8192) almost never
    # queues → locality dominates; tight capacity (32768) queues often →
    # fallback to load balancing dominates. 5s polled, but single-direction
    # safe (overload discovered late, never a false "full" on an empty replica).
    WAITING = "waiting"
    # do not do overload
    NONE = "None"
