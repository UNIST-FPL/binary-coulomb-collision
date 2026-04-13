import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_collision import Collision, MultiSpeciesCollision, Particle
from utilities import (
    canonical_three_species_case,
    canonical_three_species_weighted_case,
    particle_weight_three_species_case,
    particle_weight_two_species_case,
    simulate_relaxation,
    simulate_relaxation_multispecies,
)

DATA_DIR = PROJECT_ROOT / "tests" / "data"


def _instantiate_species(case):
    rng = np.random.default_rng(case["seed"])
    species = []
    for particle_dict in case["species"]:
        params = dict(particle_dict)
        params["rng"] = rng
        species.append(Particle(**params))
    return rng, species


def _build_payload(case):
    history = simulate_relaxation_multispecies(
        case["species"],
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )
    rng, species = _instantiate_species(case)
    collision = MultiSpeciesCollision(species, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    payload = {
        "species_names": np.asarray(history["species_names"], dtype="U32"),
        "flow_histories": history["flow_histories"],
        "flow_magnitudes": history["flow_magnitudes"],
        "temperature_histories": history["temperature_histories"],
        "time_axis": history["time_axis"],
        "reference_flow": np.array(history["reference_flow"]),
    }
    for idx, part in enumerate(species):
        payload[f"vel_{idx}"] = part.vel
    return payload


def _write_case(filename, case):
    np.savez(DATA_DIR / filename, **_build_payload(case))


def _build_two_species_particle_weight_payload(case):
    species_a, species_b = case["species"]
    history = simulate_relaxation(
        species_a,
        species_b,
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )
    rng = np.random.default_rng(case["seed"])
    particle_a = Particle(**dict(species_a, rng=rng))
    particle_b = Particle(**dict(species_b, rng=rng))
    collision = Collision(particle_a, particle_b, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    payload = {
        "species_names": np.asarray(history["species_names"], dtype="U32"),
        "flow_histories": history["flow_histories"],
        "flow_magnitudes": history["flow_magnitudes"],
        "temperature_histories": history["temperature_histories"],
        "time_axis": history["time_axis"],
        "reference_flow": np.array(history["reference_flow"]),
        "vel_a": particle_a.vel,
        "vel_b": particle_b.vel,
    }
    return payload


def _write_two_species_case(filename, case):
    np.savez(DATA_DIR / filename, **_build_two_species_particle_weight_payload(case))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_case("multispecies_3sp_small_v1.npz", canonical_three_species_case())
    _write_case("multispecies_3sp_weighted_v1.npz", canonical_three_species_weighted_case())
    _write_two_species_case("particle_weight_2sp_small_v1.npz", particle_weight_two_species_case())
    _write_case("multispecies_3sp_particle_weight_small_v1.npz", particle_weight_three_species_case())


if __name__ == "__main__":
    main()
