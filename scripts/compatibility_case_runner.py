import argparse
import json
import random

import numpy as np

from binary_collision import Collision, Particle
from binary_collision.collision import Collision as CollisionClass


def _stable_update(self) -> None:
    ops = [
        lambda: self.like_collision_update(self.spa),
        lambda: self.like_collision_update(self.spb),
        lambda: self.unlike_collision_update(),
    ]
    for op in ops:
        op()


def _load_string(payload, key: str) -> str:
    return str(payload[key].tolist())


def _load_scalar(payload, key: str):
    return payload[key].item()


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic compatibility case with the local binary_collision module.")
    parser.add_argument("--payload", required=True, help="Path to the .npz payload file.")
    args = parser.parse_args()

    payload = np.load(args.payload, allow_pickle=False)
    np.random.seed(int(_load_scalar(payload, "seed")))
    random.seed(int(_load_scalar(payload, "seed")))
    CollisionClass.update = _stable_update

    particle_1 = Particle(
        name=_load_string(payload, "particle_1_name"),
        charge=int(_load_scalar(payload, "particle_1_charge")),
        mass=float(_load_scalar(payload, "particle_1_mass")),
        density=float(_load_scalar(payload, "particle_1_density")),
        weight=float(_load_scalar(payload, "particle_1_weight")),
        Nmarker=int(_load_scalar(payload, "particle_1_nmarker")),
        vel=payload["particle_1_vel"],
    )
    particle_2 = Particle(
        name=_load_string(payload, "particle_2_name"),
        charge=int(_load_scalar(payload, "particle_2_charge")),
        mass=float(_load_scalar(payload, "particle_2_mass")),
        density=float(_load_scalar(payload, "particle_2_density")),
        weight=float(_load_scalar(payload, "particle_2_weight")),
        Nmarker=int(_load_scalar(payload, "particle_2_nmarker")),
        vel=payload["particle_2_vel"],
    )

    collision = Collision(particle_1, particle_2, dtp=float(_load_scalar(payload, "dt")))
    iterations = int(_load_scalar(payload, "iterations"))

    flow_histories = []
    temperature_histories = []
    for _ in range(iterations):
        collision.run()
        flow_histories.append(np.stack((particle_1.flow_actual.copy(), particle_2.flow_actual.copy())))
        temperature_histories.append(np.array((particle_1.temperature_actual, particle_2.temperature_actual)))

    result = {
        "case_name": _load_string(payload, "case_name"),
        "flow_histories": np.stack(flow_histories).tolist(),
        "temperature_histories": np.stack(temperature_histories).tolist(),
        "final_vel_1": particle_1.vel.tolist(),
        "final_vel_2": particle_2.vel.tolist(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
