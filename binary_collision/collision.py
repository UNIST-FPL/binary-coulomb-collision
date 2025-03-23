##############################################################################
#  Copyright 2024–2025 Fusion and Plasma application Research Laboratory (FPL), UNIST. All rights reserved.
#  Author: Sungpil YUM (sungpil.yum@unist.ac.kr)
#  This work is open source software, licensed under the terms of the
#  BSD 3-Clause License as described in the LICENSE file located in the top-level directory.
##############################################################################

from typing import Tuple
import numpy as np
import numpy.typing as npt
import warnings
import random
from scipy.constants import epsilon_0, e, physical_constants
from scipy.optimize import fsolve
from scipy import interpolate
from binary_collision.particle import Particle

warnings.simplefilter('error', RuntimeWarning)

class Collision:
    """
    Collision Class
    - Implements Coulomb collisions between multiple particle species.
    - Supports both **like-particle (same species)** and **unlike-particle (different species)** collisions.
    - K. Nanbu and S. Yonemura,
      Weighted Particles in Coulomb Collision Simulations Based on the Theory of a Cumulative Scattering Angle,
      J. Comput. Phys 145, 639 (1998)
      K. Nanbu, Theory of cumulative small-angle collisions in plasmas,
      Phys. Rev. E 55, 4642 (1997)

    Key Features:
    - Supports **two-species collisions** (can handle `spa, spb).
    - Uses **random execution order** for like-particle collisions/unlike particle collisions.
    - Supports **precomputed function tables** (`func_A_Table`) to optimize performance.
    """
    func_A_Table = None # Cached function table for faster calculations

    def __init__(self, spa: Particle, spb: Particle, dtp: float):
        """
        Initializes a collision system between two particle species.

        :param spa: First species participating in the collision.
        :param spb: Second species participating in the collision.
        :param dtp: Physical time step (Δt') used in Nanbu & Yonemura (1998).
        """
        if Collision.func_A_Table is None:
            Collision.func_A_Table = self.create_A_function()

        # Store the input order
        self._input_order = (spa, spb)

        # Assign species based on the number of markers (larger Nmarker becomes 'spa')
        self.spa, self.spb = (spa, spb) if spa.Nmarker >= spb.Nmarker else (spb, spa)
        self.spc = None  # Placeholder for like-particle processing (e.g., spc = self.spa)

        # Define time and mass parameters
        self.dtp = dtp  # System (or Physics) time ( = prime dt, i.e., \Delta t', in Nanbu98)
        self.dt = np.max([self.spa.weight, self.spb.weight]) / self.spb.weight * dtp  # Eq. (11a) in Nanbu98
        self.dt_a = self.dt
        self.dt_b = self.dt * (self.spb.weight*self.spb.Nmarker) / (self.spa.weight*self.spa.Nmarker)
        self.mu = (self.spa.mass * self.spb.mass) / (self.spa.mass + self.spb.mass)  # Reduced mass

        # Compute the number of collision events
        self.Nevent = self.spa.Nmarker  # Total collision events are determined by the larger species
        self.Nsubevent = self.spb.Nmarker # Number of events per subcycling
        self.Nsubcycling = np.ceil(self.Nevent / self.Nsubevent).astype(int)

    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: velocities in the order (original_spa, original_spb).
        """
        # Return in the original input order
        if self._input_order == (self.spa, self.spb):
            return self.spa.vel, self.spb.vel
        else:
            return self.spb.vel, self.spa.vel

    def run(self) -> None:
        """
        Runs the collision simulation by executing `update()`.
        """
        self.update()

    def update(self) -> None:
        """
        Updates all collisions in **randomized order** for a more realistic simulation.
        :return: None
        """
        # Define all collision operations (like & unlike)
        ops = [
            lambda: self.like_collision_update(self.spa),
            lambda: self.like_collision_update(self.spb),
            lambda: self.unlike_collision_update()
        ]

        # Randomize execution order
        random.shuffle(ops)

        # Execute randomized operations
        for op in ops:
            op()

    def like_collision_update(self, spc: Particle) -> None:
        """
        Simulates like-particle collisions (same species).
        :param spc: Particle species undergoing self-collisions.
        :return: None
        """
        # Shuffle velocities to ensure randomized selection
        shuffled_vel, row_map = Collision.shuffle_rows_with_map(spc.vel)

        # Split the particle group into two subgroups
        N_1, N_2 = spc.Nmarker // 2, spc.Nmarker - (spc.Nmarker // 2)
        spc_1 = Particle(name=f"{spc.name}_1", charge=spc.charge/e, mass=spc.mass/physical_constants['atomic mass constant'][0], density=spc.density, weight=spc.weight, Nmarker=N_1,
                           vel=shuffled_vel[0:N_1])
        spc_2 = Particle(name=f"{spc.name}_2", charge=spc.charge/e, mass=spc.mass/physical_constants['atomic mass constant'][0], density=spc.density, weight=spc.weight, Nmarker=N_2,
                           vel=shuffled_vel[N_1:N_1 + N_2])

        # Create a new collision object for like-particle interaction
        col_like = Collision(spc_1, spc_2, self.dtp)
        col_like.get_vstar() # Perform velocity updates

        # Update the original velocity array
        spc.vel = np.concatenate((spc_1.vel, spc_2.vel), axis=0)[np.argsort(row_map)]


    def unlike_collision_update(self) -> None:
        """
        Simulates unlike-particle collisions (different species).
        :return: None
        """
        self.get_vstar()

    def get_vstar(self) -> None:
        """
        Eq. (10a) and (10b) in "3.2. Different weight for different species" of Nanbu98
        :return: None
        """
        w_max = np.max([self.spa.weight, self.spb.weight])

        # Shuffle velocity arrays
        shuffled_vel, row_map = Collision.shuffle_rows_with_map(self.spa.vel)
        self.spa.vel = shuffled_vel
        vstar_a = np.empty((0, 3))

        # Iterate through all collision subcycling events
        for icycle in range(self.Nsubcycling):
            start = icycle * self.Nsubevent
            end = min((icycle + 1) * self.Nsubevent, self.Nevent)
            if start >= self.Nevent:
                break

            indices_a = np.arange(start, end)
            indices_b = np.arange(indices_a.size)

            # Compute new velocities using `get_vPrime`
            vp_a, vp_b = self.get_vPrime(idx_a = indices_a, idx_b = indices_b)

            # Update velocities based on statistical weights
            vp_a_true = np.where( np.random.rand(indices_a.size,1) < self.spb.weight / w_max,
                                vp_a, self.spa.vel[indices_a]) # true: prime, false: original
            vp_b_true = np.where( np.random.rand(indices_a.size,1) < self.spa.weight / w_max,
                                vp_b, self.spb.vel[indices_b])  # true: prime, false: original

            vstar_a = np.concatenate((vstar_a, vp_a_true), axis=0)
            self.spb.vel[indices_b] = vp_b_true
            self.spb.vel = self.spb.vel

        # Assign updated velocities
        self.spa.vel = vstar_a[np.argsort(row_map)]
        self.spb.vel = self.spb.vel

    def get_vPrime(self, idx_a: np.ndarray = None, idx_b: np.ndarray = None) -> \
            Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Computes post-collision velocities based on Nanbu & Yonemura (1998), Eq. (1a) & (1b).
        :param idx_a: Index array for species `spa`.
        :param idx_b: Index array for species `spb`.
        :return: Updated velocities for both species after collision (v'_a, v'_b).
        """
        va = self.spa.vel[idx_a]
        vb = self.spb.vel[idx_b]
        ma = self.spa.mass
        mb = self.spb.mass

        assert (va.shape == vb.shape)
        assert (va.shape[1] == 3)  # Ensure velocity components are (vx, vy, vz)

        mab: float = ma + mb # Total mass of the colliding pair
        g: npt.NDArray[float] = va - vb # Relative velocity
        h: npt.NDArray[float] = self.get_h(g) # Generate random orthogonal vector to `g`

        # Compute cumulative scattering parameter s_ab and s_ba
        s_ab = self.evaluate_s_ab(g, self.dt_a, self.spb.density)
        s_ba = s_ab * (self.spa.weight * self.spa.Nmarker * self.dt_b) / (self.spb.weight * self.spb.Nmarker * self.dt_a)
        # Note that s_ba = s_ba * n_a / n_b from Nanbu & Yonemura 1998 eq(7)
        # Since the collision occurs in the same volume, density = Nmarker * weight (as utilized in Nanbu98)
        # self.s_ba = self.s_ab * (spa.weight*spa.Nmarker) / (spb.weight*spb.Nmarker)

        cosChi_ab, cosChi_ba = self.get_cosChi(s_ab, s_ba)

        try:
            vPrime_a = va - mb / mab * (g * (1 - cosChi_ab) + h * np.sqrt((1 - cosChi_ab ** 2)))
        except RuntimeWarning:
            print("RuntimeWarning from cosChi_ab!")
            print("rand_array =", s_ab)
            raise
        try:
            vPrime_b = vb + ma / mab * (g * (1 - cosChi_ba) + h * np.sqrt((1 - cosChi_ba ** 2)))
        except RuntimeWarning:
            print("RuntimeWarning from cosChi_ba!")
            print("rand_array =", s_ba)
            raise
        return vPrime_a, vPrime_b

    def get_cosChi(self, s_ab, s_ba) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Computes the cosine of the scattering angles Chi_ab and Chi_ba based on Nanbu (1997), Eq. (17).
        :param s_ab: Cumulative scattering parameter for species `spa` to `spb` based on Nanbu & Yonemura (1998), Eq. (4).
        :param s_ba: Cumulative scattering parameter for species `spb` to `spa` based on Nanbu & Yonemura (1998), Eq. (4).
        :return: Cosine of the scattering angles (cosChi_ab, cosChi_ba).
        """
        cosChi_ab, cosChi_ba = self.evaluate_cosChi(s_ab, s_ba)
        return cosChi_ab, cosChi_ba

    def evaluate_cosChi(self, s_ab, s_ba)-> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Computes `cos(Chi)` based on precomputed values of `A`.

        - Implements the statistical method for small and large `s` values to avoid numerical instability.
        - See Eq. (17) in Nanbu (1997) for details.

        :param s_ab: Scattering parameter for `spa -> spb`.
        :param s_ba: Scattering parameter for `spb -> spa`.
        :return: Cosine of the scattering angles (cosChi_ab, cosChi_ba).
        """
        A_ab = np.clip(self.get_A(s_ab), 1e-12, 400.)
        A_ba = np.clip(self.get_A(s_ba), 1e-12, 400.)
        rand_array = np.random.rand(*np.shape(A_ab))

        mask_small_ab = (s_ab < 1.e-4)
        mask_large_ab = (s_ab > 6.0)

        mask_small_ba = (s_ba < 1.e-4)
        mask_large_ba = (s_ba > 6.0)

        cosChi_ab = np.log(np.exp(-A_ab) + 2. * rand_array * np.sinh(A_ab)) / A_ab
        cosChi_ba = np.log(np.exp(-A_ba) + 2. * rand_array * np.sinh(A_ba)) / A_ba

        # From NANBU97 eq(17), small s, large s are treated as below to avoid overflow
        cosChi_ab[mask_small_ab] = 1.0 + s_ab[mask_small_ab] * np.log(rand_array[mask_small_ab])
        cosChi_ba[mask_small_ba] = 1.0 + s_ba[mask_small_ba] * np.log(rand_array[mask_small_ba])
        cosChi_ab[mask_large_ab] = 2.0 * rand_array[mask_large_ab] - 1.0
        cosChi_ba[mask_large_ba] = 2.0 * rand_array[mask_large_ba] - 1.0

        return cosChi_ab[:, np.newaxis], cosChi_ba[:, np.newaxis]

    def get_A(self, s):
        """
        Retrieves or computes the `A` parameter used in the scattering function.

        :param s: Scattering parameter.
        :return: `A` value computed using a precomputed function table.
        """
        s = np.asarray(s)
        result = np.where(s < 0.01, 1. / s,
                          np.where(s > 3., 3. * np.exp(-s), self.func_A_Table(s)))
        return result

    @classmethod
    def create_A_function(self):
        """
        Creates a precomputed function table for `A(s)`.
        - Uses interpolation to improve performance.
        :return: `A(s)` function table.
        """
        s_sample = np.append(np.linspace(0.01, 0.1, 10), np.linspace(0.2, 5.0, 49))
        A_table = self.solve_A_fsolve(s_sample)
        f = interpolate.interp1d(s_sample, A_table, kind='linear', bounds_error=False)
        return f

    @classmethod
    def solve_A_fsolve(self, s, initial_guess=10.):
        """
        Computes `A(s)` using numerical solving (`fsolve`).
        - Eq. (3) in Nanbu (1998).

        :param s: Scattering parameter.
        :param initial_guess: Initial guess for numerical solver.
        :return: Solved `A(s)` values.
        """
        ig_arr = np.full(s.shape, initial_guess)
        def eq_A(A, s):
            return np.cosh(A)/np.sinh(A) - 1./A - np.exp(-s)
        A_solution = fsolve(eq_A, ig_arr, args=(s,))
        return A_solution

    def evaluate_s_ab(self, g, dt_a, density_b)  -> npt.NDArray[float]:
        """
        Computes the scattering parameter `s_ab from Eq. 4 in Nanbu & Yonemura 1998.

        :param g: Relative velocity.
        :param dt_a: Time step.
        :param density_b: Number density of species `b`.
        :return: `s_ab` values.
        """
        return self.lnLambda_ab() \
               * (self.spa.charge * self.spb.charge / (epsilon_0 * self.mu)) ** 2 \
               * density_b * dt_a * 0.25 / np.pi \
               * np.linalg.norm(g, axis=1) ** -3


    def lnLambda_ab(self) -> float:
        """
        Computes the Coulomb logarithm `ln(Λ)`.
        return: ln(Λ)
        """
        return np.log(self.Lambda_ab())

    def Lambda_ab(self) -> float:
        """
        Computes the Coulomb logarithm factor `Λ`.
        return: Λ value
        """
        return 2. * np.pi * epsilon_0 * self.mu * self.debye() * self.g2_ab() / np.abs(self.spa.charge * self.spb.charge)

    def debye(self) -> float:
        """
        Debye length between species 'a' and 'b' only. Not for all species. Below is a general equation for all species.

        lambda_D = \\sqrt{ \frac{epsilon_0}{\\sum_s n_s q_s^2 / (T_s e)} }

        , where 's' is species index for both ions and electrons, 'z' is the charge number, and T_s is in eV unit.
        This could be under the Particle class, but can be only relevant to collision operations for some applications.
        :return: Debye-Huckel equation ; Symmetry in species
        """
        return np.sqrt( epsilon_0 /
                        (
                                self.spa.density * self.spa.charge**2 / (self.spa.temperature_actual * e)
                                + self.spb.density * self.spb.charge**2 / (self.spb.temperature_actual * e)
                          )
                        )
    def g2_ab(self) -> float:
        """
        Computes the squared characteristic relative velocity g²_ab
        between two species (a and b), including both thermal and flow contributions.

        :return: Squared relative velocity (g²_ab) in units of m²/s² as a float64 ; Symmetry in species
        """
        return 3. * e * ( self.spa.temperature_actual / self.spa.mass + self.spb.temperature_actual / self.spb.mass ) \
             + np.linalg.norm(self.spa.flow_actual - self.spb.flow_actual)**2

    @staticmethod
    def get_h(g: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Generates a random unit vector `h` that is orthogonal to the relative velocity `g`.

        This method is used in the Nanbu (1997) binary collision model to define the scattering plane.
        The resulting `h` vector lies in the plane perpendicular to `g`, with a uniformly random angle.

        :param g: Relative velocity vectors between particles, shape (N, 3)
                  Each row corresponds to a pairwise difference v_a - v_b.
        :return: Array of orthogonal vectors `h` with shape (N, 3), one for each g vector.
        """
        N = g.shape[0] # Number of particle pairs

        # Decompose the g vector into components
        gx: npt.NDArray[float] = g[:, 0]
        gy: npt.NDArray[float] = g[:, 1]
        gz: npt.NDArray[float] = g[:, 2]

        # Compute g_perp = sqrt(gy² + gz²)
        g_perp: npt.NDArray[float] = gy * gy + gz * gz  # Perpendicular squared component
        g_abs: npt.NDArray[float] = np.sqrt(gx * gx + g_perp) # Magnitude of g
        g_perp = np.sqrt(g_perp) # Magnitude of perpendicular component

        # Random azimuthal angle (ϵ) between 0 and 2π for isotropic scattering
        ep: npt.NDArray[float] = np.random.rand(N) * np.pi * 2.
        cos_ep: npt.NDArray[float] = np.cos(ep)
        sin_ep: npt.NDArray[float] = np.sin(ep)

        # Compute components of orthogonal vector h (see Nanbu 1997 derivation)
        hx: npt.NDArray[float] = g_perp * cos_ep
        gx_cosep: npt.NDArray[float] = gx * cos_ep
        gabs_sinep: npt.NDArray[float] = g_abs * sin_ep
        hy: npt.NDArray[float] = - (gy*gx_cosep + gz*gabs_sinep) / g_perp
        hz: npt.NDArray[float] = (gy*gabs_sinep - gz*gx_cosep) / g_perp

        # Return stacked orthogonal vectors
        return np.stack((hx, hy, hz), axis=1)

    @staticmethod
    def shuffle_rows_with_map(array: np.ndarray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Shuffles the rows of an array and returns the shuffled array with a mapping index.

        :param array: Input array.
        :return: (Shuffled array, row mapping)
        """
        row_map = np.random.permutation(array.shape[0])
        shuffled_array = array[row_map,:]
        return shuffled_array, row_map
