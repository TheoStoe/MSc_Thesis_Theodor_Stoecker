"""Active-cohort exit and reserve-refill policies."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..contracts import UserId
from ..utility.fingerprints import fingerprint_semantic_state
from ._common import (
    iteration as validate_iteration,
    policy_fingerprint,
    positive_id,
    unique_ids,
)
from ._population_refill import (
    select_survivor_similar_reserve,
    select_uniform_reserve,
)
from .contracts import PopulationPolicy, PopulationTransitionResult


class _PopulationPolicy:
    _variants = {
        "fixed": "fixed_active_users_v1",
        "exit": "exit_without_refill_v1",
        "uniform": "uniform_reserve_refill_v1",
        "survivor": "survivor_similar_reserve_refill_v1",
    }
    supported_survivor_rules = {
        "survivor_cluster_weighted_uniform_leave_unfilled_v1",
        "survivor_cluster_weighted_uniform_global_fallback_v1",
    }

    def __init__(
        self,
        *,
        mode: str,
        initial_active_user_ids: Sequence[UserId],
        reserve_user_ids: Sequence[UserId] = (),
        seed: int = 0,
        cluster_by_user: Mapping[UserId, int] | None = None,
        similarity_artifact_fingerprint: str | None = None,
        selection_rule_id: str | None = None,
    ) -> None:
        if mode not in self._variants:
            raise ValueError(f"unknown population-policy mode: {mode}")
        self.variant_id = self._variants[mode]
        self._mode = mode
        self._active = list(unique_ids(initial_active_user_ids, "active users"))
        self._reserve = list(
            unique_ids(reserve_user_ids, "reserve users", allow_empty=True)
        )
        if set(self._active).intersection(self._reserve):
            raise ValueError("active and reserve users must be disjoint")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._seed = seed
        self._initial_active = frozenset(self._active)
        self._initial_reserve = frozenset(self._reserve)
        self._all_users = self._initial_active.union(self._initial_reserve)
        self._exited_user_ids: set[UserId] = set()
        self._activated_user_ids: set[UserId] = set()
        self._scheduled_exits: dict[UserId, int] = {}
        self._last_applied_iteration = 0
        self._similarity_artifact_fingerprint = similarity_artifact_fingerprint
        self._selection_rule_id = selection_rule_id
        self._clusters: dict[UserId, int] = {}

        if mode == "survivor":
            if selection_rule_id not in self.supported_survivor_rules:
                raise ValueError(
                    "selection_rule_id must name one supported survivor rule"
                )
            if (
                not isinstance(similarity_artifact_fingerprint, str)
                or not similarity_artifact_fingerprint.strip()
            ):
                raise ValueError("similarity_artifact_fingerprint cannot be blank")
            if cluster_by_user is None:
                raise ValueError("survivor refill requires frozen clusters")
            if set(cluster_by_user) != set(self._all_users):
                raise ValueError(
                    "cluster coverage must equal active plus reserve users"
                )
            for user_id in sorted(self._all_users):
                cluster = cluster_by_user[user_id]
                if isinstance(cluster, bool) or not isinstance(cluster, int):
                    raise ValueError("cluster IDs must be integers")
                self._clusters[user_id] = cluster
        elif any(
            value is not None
            for value in (
                cluster_by_user,
                similarity_artifact_fingerprint,
                selection_rule_id,
            )
        ):
            raise ValueError("similarity settings belong only to survivor refill")

    def apply_scheduled_transitions(
        self, iteration: int
    ) -> PopulationTransitionResult:
        iteration = validate_iteration(iteration)
        if iteration != self._last_applied_iteration + 1:
            raise ValueError("population transitions must be applied once in order")
        overdue = [
            user_id
            for user_id, effective in self._scheduled_exits.items()
            if effective < iteration
        ]
        if overdue:
            raise RuntimeError(f"missed scheduled exits: {sorted(overdue)}")
        exiting = tuple(
            user_id
            for user_id in self._active
            if self._scheduled_exits.get(user_id) == iteration
        )
        exit_set = set(exiting)
        self._active = [
            user_id for user_id in self._active if user_id not in exit_set
        ]
        self._exited_user_ids.update(exiting)
        for user_id in exiting:
            del self._scheduled_exits[user_id]

        activated: list[UserId] = []
        selection_records: list[Mapping[str, Any]] = []
        survivors = tuple(self._active)
        if self._mode in {"uniform", "survivor"}:
            for slot, _exited_user_id in enumerate(exiting):
                if self._mode == "uniform":
                    entrant, record = select_uniform_reserve(
                        reserve=self._reserve,
                        seed=self._seed,
                        iteration=iteration,
                        slot=slot,
                    )
                else:
                    assert self._selection_rule_id is not None
                    entrant, record = select_survivor_similar_reserve(
                        reserve=self._reserve,
                        seed=self._seed,
                        iteration=iteration,
                        slot=slot,
                        survivors=survivors,
                        clusters=self._clusters,
                        selection_rule_id=self._selection_rule_id,
                    )
                selection_records.append(record)
                if entrant is not None:
                    activated.append(entrant)
                    self._activated_user_ids.add(entrant)
            self._active.extend(activated)

        self._last_applied_iteration = iteration
        active = tuple(self._active)
        risk_fingerprint = fingerprint_semantic_state(
            {"active_user_ids": active}, "active-risk-set-v1"
        )
        return PopulationTransitionResult(
            iteration=iteration,
            exited_user_ids=exiting,
            activated_user_ids=tuple(activated),
            active_user_ids=active,
            risk_set_fingerprint=risk_fingerprint,
            metadata=MappingProxyType(
                {
                    "policy_variant": self.variant_id,
                    "requested_refills": (
                        len(exiting)
                        if self._mode in {"uniform", "survivor"}
                        else 0
                    ),
                    "realized_refills": len(activated),
                    "reserve_remaining": len(self._reserve),
                    "selections": tuple(
                        MappingProxyType(dict(record))
                        for record in selection_records
                    ),
                }
            ),
        )

    def schedule_exit(self, user_id: UserId, effective_iteration: int) -> None:
        if self._mode == "fixed":
            raise ValueError("fixed population does not permit exits")
        user_id = positive_id(user_id, "user_id")
        effective_iteration = validate_iteration(effective_iteration)
        if user_id not in self._active:
            raise ValueError(f"cannot exit inactive user: {user_id}")
        if effective_iteration != self._last_applied_iteration + 1:
            raise ValueError("exit must become effective at t+1")
        previous = self._scheduled_exits.get(user_id)
        if previous is not None and previous != effective_iteration:
            raise ValueError("user already has a different scheduled exit")
        self._scheduled_exits[user_id] = effective_iteration

    def export_state(self) -> Mapping[str, Any]:
        return {
            "variant_id": self.variant_id,
            "mode": self._mode,
            "seed": self._seed,
            "similarity_artifact_fingerprint": (
                self._similarity_artifact_fingerprint
            ),
            "selection_rule_id": self._selection_rule_id,
            "active_user_ids": tuple(self._active),
            "reserve_user_ids": tuple(self._reserve),
            "exited_user_ids": tuple(sorted(self._exited_user_ids)),
            "activated_user_ids": tuple(sorted(self._activated_user_ids)),
            "scheduled_exits": tuple(sorted(self._scheduled_exits.items())),
            "last_applied_iteration": self._last_applied_iteration,
        }

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("population-policy state must be a mapping")
        expected_config = {
            "variant_id": self.variant_id,
            "mode": self._mode,
            "seed": self._seed,
            "similarity_artifact_fingerprint": (
                self._similarity_artifact_fingerprint
            ),
            "selection_rule_id": self._selection_rule_id,
        }
        if any(state.get(key) != value for key, value in expected_config.items()):
            raise ValueError("population-policy state/configuration mismatch")
        active = list(
            unique_ids(
                state.get("active_user_ids", ()),
                "active users",
                allow_empty=True,
            )
        )
        reserve = list(
            unique_ids(
                state.get("reserve_user_ids", ()),
                "reserve users",
                allow_empty=True,
            )
        )
        exited = set(
            unique_ids(
                state.get("exited_user_ids", ()),
                "exited users",
                allow_empty=True,
            )
        )
        activated = set(
            unique_ids(
                state.get("activated_user_ids", ()),
                "activated users",
                allow_empty=True,
            )
        )
        partitions = (set(active), set(reserve), exited)
        if any(
            partitions[left].intersection(partitions[right])
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        ):
            raise ValueError("restored population partitions overlap")
        if set(active).union(reserve, exited) != self._all_users:
            raise ValueError("restored population does not preserve every user")
        if not activated.issubset(self._initial_reserve):
            raise ValueError("only reserve users can have activation history")
        if self._initial_reserve != set(reserve).union(activated):
            raise ValueError("restored reserve/activation provenance is invalid")
        if set(active).intersection(self._initial_reserve) != activated.difference(
            exited
        ):
            raise ValueError(
                "restored active reserve users lack activation provenance"
            )
        if self._initial_active != set(active).intersection(
            self._initial_active
        ).union(exited.intersection(self._initial_active)):
            raise ValueError("invalid restored population membership")
        last = state.get("last_applied_iteration")
        if isinstance(last, bool) or not isinstance(last, int) or last < 0:
            raise ValueError("invalid restored population iteration")
        scheduled_raw = state.get("scheduled_exits", ())
        scheduled: dict[UserId, int] = {}
        for record in scheduled_raw:
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                raise ValueError("invalid restored scheduled exit")
            user_id = positive_id(record[0], "scheduled user_id")
            effective = validate_iteration(record[1])
            if (
                user_id not in active
                or effective != last + 1
                or user_id in scheduled
            ):
                raise ValueError("invalid restored scheduled exit")
            scheduled[user_id] = effective
        self._active = active
        self._reserve = reserve
        self._exited_user_ids = exited
        self._activated_user_ids = activated
        self._scheduled_exits = scheduled
        self._last_applied_iteration = last

    def fingerprint(self) -> str:
        semantic_state = dict(self.export_state())
        # The cursor is needed for resume but is not a population mutation.
        semantic_state.pop("last_applied_iteration")
        return policy_fingerprint(self.variant_id, semantic_state)


def build_fixed_active_population_policy(
    *, initial_active_user_ids: Sequence[UserId]
) -> PopulationPolicy:
    """Build a fixed cohort; scheduling an exit is a wiring error."""

    return _PopulationPolicy(
        mode="fixed", initial_active_user_ids=initial_active_user_ids
    )


def build_exit_without_refill_population_policy(
    *, initial_active_user_ids: Sequence[UserId]
) -> PopulationPolicy:
    """Build the exit-only ablation, with exits effective at t+1."""

    return _PopulationPolicy(
        mode="exit", initial_active_user_ids=initial_active_user_ids
    )


def build_uniform_reserve_refill_population_policy(
    *,
    initial_active_user_ids: Sequence[UserId],
    reserve_user_ids: Sequence[UserId],
    seed: int,
) -> PopulationPolicy:
    """Build one-for-one uniform reserve refill without replacement."""

    return _PopulationPolicy(
        mode="uniform",
        initial_active_user_ids=initial_active_user_ids,
        reserve_user_ids=reserve_user_ids,
        seed=seed,
    )


def build_survivor_similar_reserve_refill_population_policy(
    *,
    initial_active_user_ids: Sequence[UserId],
    reserve_user_ids: Sequence[UserId],
    cluster_by_user: Mapping[UserId, int],
    similarity_artifact_fingerprint: str,
    selection_rule_id: str,
    seed: int,
) -> PopulationPolicy:
    """Build L4 refill by survivor-cluster share and uniform within-cluster draw."""

    return _PopulationPolicy(
        mode="survivor",
        initial_active_user_ids=initial_active_user_ids,
        reserve_user_ids=reserve_user_ids,
        seed=seed,
        cluster_by_user=cluster_by_user,
        similarity_artifact_fingerprint=similarity_artifact_fingerprint,
        selection_rule_id=selection_rule_id,
    )


__all__ = [
    "build_exit_without_refill_population_policy",
    "build_fixed_active_population_policy",
    "build_survivor_similar_reserve_refill_population_policy",
    "build_uniform_reserve_refill_population_policy",
]
