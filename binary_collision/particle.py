##############################################################################
#  Copyright 2024–2025 Fusion and Plasma application Research Laboratory (FPL), UNIST. All rights reserved.
#  Author: Sungpil YUM (sungpil.yum@unist.ac.kr)
#  This work is open source software, licensed under the terms of the
#  BSD 3-Clause License as described in the LICENSE file located in the top-level directory.
##############################################################################

import numpy as np
import numpy.typing as npt
from scipy.constants import epsilon_0, e, physical_constants

class Particle:
    """
    Particle Class
    - Represents an individual particle.
    - Stores physical properties.

    Key Features:
    - Users can provide a velocity array (`vel`) or initialize isotropic velocity using Maxwellian distribution.
    - Automatically computes **actual flow (`flow_actual`)** and **temperature (`temperature_actual`)**.
    - Uses `@property` to update flow and temperature when velocity changes.
    """
    def __init__(self, name: str, charge: int, mass: float, density: float,
                       flow: float = None, temperature: float = None, weight: float = 1.0, Nmarker: int = 1,
                       vel: npt.NDArray[float] = None):
        """
        Initializes the Particle object with fundamental physical properties.

        :param name: Species name (e.g., "D+", "e-").
        :param charge: Charge of the particle (in electron units). It is converted to Coulombs.
        :param mass: Mass of the particle (given in atomic mass units, "u"). Converted to kg.
        :param density: Number density of the species (particles per cubic meter, /m³).
        :param flow: Initial flow velocity (m/s). Required for Maxwellian distribution initialization.
        :param temperature: Initial temperature (eV). Required for Maxwellian distribution initialization.
        :param weight: Weight factor.
        :param Nmarker: Number of marker particles.
        :param vel: Optional velocity array (`shape=(Nmarker, 3)`).
                    If not provided, Maxwellian initialization is used.
        """

        # Particle properties
        self.name: str = name # Particle species name
        self.charge: float = charge * e  # Convert charge to Coulombs
        self.mass: float = mass * physical_constants['atomic mass constant'][0]  # Convert mass to kg
        self.density: float = density  # Particle density (particles per m³)
        self.flow_given: float = flow   # Given flow velocity (m/s)
        self.temperature_given: float = temperature  # Given temperature (eV)
        self.weight: float = weight # Weight factor
        self.Nmarker: int = Nmarker # Number of marker particles

        # Velocity and computed properties
        self._vel = None # Private velocity attribute
        self.flow_actual = None # Actual computed flow velocity
        self.temperature_actual = None # Actual computed temperature

        # Initialize velocity: Use provided `vel` or generate isotropic velocity by Maxwellian distribution
        if vel is not None and isinstance(vel, np.ndarray) and vel.shape == (Nmarker, 3):
            self.vel = vel # Assign provided velocity
        else:
            print(f"{self.name}: velocity is set as isotropic.")
            self.set_vel_isotropic() # Initialize isotropic velocity using Maxwellian distribution

    @property
    def vel(self):
        """
        Getter for velocity (`vel`).
        - Returns the private velocity `_vel`.
        """
        return self._vel

    @vel.setter
    def vel(self, data):
        """
        Setter for velocity (`vel`).
        - Automatically updates **actual flow** and **actual temperature** whenever velocity changes.
        """
        self._vel = data # Assign new velocity
        self.set_flow_actual() # Compute actual flow velocity
        self.set_temperature_actual() # Compute actual temperature

    def set_flow_actual(self) -> None:
        """
        Compute the actual flow velocity (`flow_actual`).
        - Takes the **mean velocity** of all particles as the flow velocity.
        """
        assert self.vel is not None, "Velocity must be set before computing flow."
        self.flow_actual = np.mean(self.vel, axis=0) # Compute mean velocity
        assert self.flow_actual.size == 3  # Ensure it has (vx, vy, vz) components

    def set_temperature_actual(self) -> None:
        """
        Compute the actual temperature (`temperature_actual`).
        - Assumes a Maxwellian distribution and calculates temperature based on the mean squared velocity.
        - Formula: T_actual = (m / (3 * e)) * (⟨v²⟩ - ⟨flow²⟩)
        """
        assert self.vel is not None, "Velocity must be set before computing temperature."
        assert self.flow_actual is not None, "Flow must be computed before temperature."
        self.temperature_actual = (self.mass / (3. * e)
                                * (np.sum(np.square(self.vel))/self.Nmarker - sum(np.square(self.flow_actual))))

    def set_vel_isotropic(self) -> None:
        """
        Initialize velocity using an **isotropic Maxwellian distribution**.
        - Uses the given `flow_given` and `temperature_given` to generate a velocity distribution.
        - Each velocity component (vx, vy, vz) is sampled **independently** from a Gaussian.
        """
        # Compute isotropic flow vector
        U = self.flow_given # Given flow velocity
        flow_vec = np.array([U/np.sqrt(3), U/np.sqrt(3), U/np.sqrt(3)], dtype=float) # Isotropic assumption
        assert(self.Nmarker)
        # Generate random Maxwellian-distributed velocities
        T = self.temperature_given # Given temperature
        self.vel = np.random.normal(flow_vec, np.sqrt(T * e / self.mass), (self.Nmarker, 3))