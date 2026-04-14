from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .collision import Collision
from .particle import Particle, RNGLike, _coerce_rng


@dataclass(frozen=True)
class CollisionStage:
    kind: str
    species_i: int
    species_j: int


class MultiSpeciesCollision:
    """
    Orchestrate Nanbu pairwise operators across 3+ species without changing the
    existing two-species `Collision` API.
    """

    def __init__(self, species: Iterable[Particle], dtp: float, rng: RNGLike = None):
        self.species = tuple(species)
        if len(self.species) < 2:
            raise ValueError("MultiSpeciesCollision requires at least two species.")

        self._input_order = self.species
        self.dtp = dtp
        self.rng = _coerce_rng(rng)
        self._pair_collisions: dict[tuple[int, int], Collision] = {}
        self._like_collisions: dict[int, Collision] = {}
        self._plasma_species_context = self.species if len(self.species) > 2 else None
        self._stage_catalog = self._build_stage_catalog()

    def get_velocity(self):
        return tuple(species.vel for species in self._input_order)

    def get_velocities(self):
        return self.get_velocity()

    def run(self) -> None:
        for stage in self._ordered_stages():
            self._run_stage(stage)

    def _build_stage_catalog(self) -> tuple[CollisionStage, ...]:
        stages: list[CollisionStage] = []
        for species_i in range(len(self.species)):
            stages.append(CollisionStage("like", species_i, species_i))
        for species_i in range(len(self.species)):
            for species_j in range(species_i + 1, len(self.species)):
                stages.append(CollisionStage("unlike", species_i, species_j))
        return tuple(stages)

    def _ordered_stages(self) -> tuple[CollisionStage, ...]:
        order = self._permutation(len(self._stage_catalog))
        return tuple(self._stage_catalog[idx] for idx in order)

    def _run_stage(self, stage: CollisionStage) -> None:
        if stage.kind == "like":
            collision = self._get_like_collision(stage.species_i)
            collision.like_collision_update(self.species[stage.species_i])
            return

        collision = self._get_pair_collision(stage.species_i, stage.species_j)
        collision.unlike_collision_update()

    def _get_pair_collision(self, species_i: int, species_j: int) -> Collision:
        key = (min(species_i, species_j), max(species_i, species_j))
        collision = self._pair_collisions.get(key)
        if collision is None:
            collision = Collision(
                self.species[key[0]],
                self.species[key[1]],
                self.dtp,
                rng=self.rng,
                plasma_species=self._plasma_species_context,
            )
            self._pair_collisions[key] = collision
        return collision

    def _get_like_collision(self, species_i: int) -> Collision:
        collision = self._like_collisions.get(species_i)
        if collision is None:
            species = self.species[species_i]
            collision = Collision(
                species,
                species,
                self.dtp,
                rng=self.rng,
                plasma_species=self._plasma_species_context,
            )
            self._like_collisions[species_i] = collision
        return collision

    def _permutation(self, size: int) -> np.ndarray:
        if self.rng is None:
            return np.random.permutation(size)
        return self.rng.permutation(size)
