"""Small NumPy RNG factory with explicit, call-order-independent coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import UserId


@dataclass(frozen=True)
class RandomKey:
    """Coordinates for one deterministic stochastic decision."""

    stream_id: int
    user_id: UserId | None = None
    iteration: int | None = None
    index: int | None = None


def random_generator(root_seed: int, key: RandomKey) -> np.random.Generator:
    """Create an ordinary ``np.random`` generator for one explicit decision.

    ``SeedSequence`` mixes the root seed and integer coordinates. This keeps the
    implementation simple while preventing one user's extra call or a worker-
    ordering change from advancing another user's draw.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer")
    if root_seed < 0:
        raise ValueError("root_seed cannot be negative")
    if not isinstance(key, RandomKey):
        raise TypeError("key must be a RandomKey")
    coordinates = (key.stream_id, key.user_id, key.iteration, key.index)
    for value, name in zip(
        coordinates,
        ("stream_id", "user_id", "iteration", "index"),
        strict=True,
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None")
    # Adding one distinguishes a real zero from an absent optional coordinate.
    spawn_key = tuple(0 if value is None else value + 1 for value in coordinates)
    return np.random.default_rng(
        np.random.SeedSequence(entropy=root_seed, spawn_key=spawn_key)
    )

# Stateful third-party components must still export and restore their RNG state.
# Population sampling is a prepared artifact with its own population seed.
