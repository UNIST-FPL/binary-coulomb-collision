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
from scipy.constants import epsilon_0, e, physical_constants
from scipy.optimize import fsolve
from scipy import interpolate
from binary_collision.particle import Particle, RNGLike, _coerce_rng

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
    min_temperature_ev = 1.0e-12

    def __init__(self, spa: Particle, spb: Particle, dtp: float, rng: RNGLike = None):
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
        self.rng = _coerce_rng(rng)
        self._like_workspaces = {}

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
        ops = [ops[idx] for idx in self._permutation(len(ops))]

        # Execute randomized operations
        for op in ops:
            op()

    def like_collision_update(self, spc: Particle) -> None:
        """
        Simulates like-particle collisions (same species).
        :param spc: Particle species undergoing self-collisions.
        :return: None
        """
        if spc.Nmarker < 4:
            return

        # Shuffle velocities to ensure randomized selection
        shuffled_vel, row_map = Collision.shuffle_rows_with_map(spc.vel, rng=self.rng)

        N_1, N_2, spc_1, spc_2, col_like = self._get_like_workspace(spc, shuffled_vel)
        col_like.get_vstar() # Perform velocity updates

        # Update the original velocity array
        updated_shuffled = np.empty_like(shuffled_vel)
        updated_shuffled[:N_1] = spc_1.vel
        updated_shuffled[N_1:N_1 + N_2] = spc_2.vel
        spc.vel = Collision.restore_rows_from_map(updated_shuffled, row_map)


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
        prob_a = self.spb.weight / w_max
        prob_b = self.spa.weight / w_max

        # Shuffle velocity arrays
        shuffled_vel, row_map = Collision.shuffle_rows_with_map(self.spa.vel, rng=self.rng)
        self.spa.assign_vel(shuffled_vel, refresh_stats=True)
        vstar_a = np.empty_like(shuffled_vel)
        spb_vel = self.spb.vel

        # Iterate through all collision subcycling events
        for icycle in range(self.Nsubcycling):
            start = icycle * self.Nsubevent
            end = min((icycle + 1) * self.Nsubevent, self.Nevent)
            if start >= self.Nevent:
                break

            count = end - start
            indices_a = slice(start, end)
            indices_b = slice(0, count)

            # Compute new velocities using `get_vPrime`
            vp_a, vp_b = self.get_vPrime(idx_a = indices_a, idx_b = indices_b)
            current_a = self.spa.vel[indices_a]
            current_b = spb_vel[indices_b]

            # Update velocities based on statistical weights
            vp_a_true = np.where(self._random((count, 1)) < prob_a, vp_a, current_a) # true: prime, false: original
            vp_b_true = np.where(self._random((count, 1)) < prob_b, vp_b, current_b)  # true: prime, false: original

            vstar_a[indices_a] = vp_a_true
            spb_vel[indices_b] = vp_b_true
            self.spb.update_moments()

        # Assign updated velocities
        self.spa.vel = Collision.restore_rows_from_map(vstar_a, row_map)

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
        factor_a = mb / mab
        factor_b = ma / mab
        g: npt.NDArray[float] = va - vb # Relative velocity
        h: npt.NDArray[float] = self.get_h(g, rng=self.rng) # Generate random orthogonal vector to `g`

        # Compute cumulative scattering parameter s_ab and s_ba
        s_ab = self.evaluate_s_ab(g, self.dt_a, self.spb.density)
        s_ba = s_ab * (self.spa.weight * self.spa.Nmarker * self.dt_b) / (self.spb.weight * self.spb.Nmarker * self.dt_a)
        # Note that s_ba = s_ba * n_a / n_b from Nanbu & Yonemura 1998 eq(7)
        # Since the collision occurs in the same volume, density = Nmarker * weight (as utilized in Nanbu98)
        # self.s_ba = self.s_ab * (spa.weight*spa.Nmarker) / (spb.weight*spb.Nmarker)

        cosChi_ab, cosChi_ba = self.get_cosChi(s_ab, s_ba)

        try:
            vPrime_a = va - factor_a * (g * (1 - cosChi_ab) + h * np.sqrt((1 - cosChi_ab ** 2)))
        except RuntimeWarning:
            print("RuntimeWarning from cosChi_ab!")
            print("rand_array =", s_ab)
            raise
        try:
            vPrime_b = vb + factor_b * (g * (1 - cosChi_ba) + h * np.sqrt((1 - cosChi_ba ** 2)))
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
        rand_array = self._random(np.shape(A_ab))
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
        result = np.empty_like(s, dtype=float)
        small_mask = s < 0.01
        large_mask = s > 3.0
        mid_mask = ~(small_mask | large_mask)

        if np.any(small_mask):
            small_values = s[small_mask]
            small_result = np.empty_like(small_values, dtype=float)
            positive_small = small_values > 0.0
            small_result[positive_small] = 1.0 / small_values[positive_small]
            small_result[~positive_small] = np.inf
            result[small_mask] = small_result
        result[large_mask] = 3.0 * np.exp(-s[large_mask])
        if np.any(mid_mask):
            result[mid_mask] = self.func_A_Table(s[mid_mask])
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
        g_norm = np.linalg.norm(g, axis=1)
        s_ab = np.zeros_like(g_norm)
        active = g_norm > 0.0
        if np.any(active):
            prefactor = self.lnLambda_ab() \
                       * (self.spa.charge * self.spb.charge / (epsilon_0 * self.mu)) ** 2 \
                       * density_b * dt_a * 0.25 / np.pi
            s_ab[active] = prefactor * g_norm[active] ** -3
        return s_ab


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
        temp_a = self._effective_temperature(self.spa)
        temp_b = self._effective_temperature(self.spb)
        return np.sqrt( epsilon_0 /
                        (
                                self.spa.density * self.spa.charge**2 / (temp_a * e)
                                + self.spb.density * self.spb.charge**2 / (temp_b * e)
                          )
                        )
    def g2_ab(self) -> float:
        """
        Computes the squared characteristic relative velocity g²_ab
        between two species (a and b), including both thermal and flow contributions.

        :return: Squared relative velocity (g²_ab) in units of m²/s² as a float64 ; Symmetry in species
        """
        return 3. * e * ( self._effective_temperature(self.spa) / self.spa.mass + self._effective_temperature(self.spb) / self.spb.mass ) \
             + np.linalg.norm(self.spa.flow_actual - self.spb.flow_actual)**2

    @staticmethod
    def get_h(g: npt.NDArray[float], rng: RNGLike = None) -> npt.NDArray[float]:
        """
        Generates a random vector `h` that is orthogonal to the relative velocity `g`.

        This method is used in the Nanbu (1997) binary collision model to define the scattering plane.
        The resulting `h` vector lies in the plane perpendicular to `g`, with a uniformly random angle
        and the same magnitude as `g`.

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
        g_perp_sq: npt.NDArray[float] = gy * gy + gz * gz  # Perpendicular squared component
        g_abs: npt.NDArray[float] = np.sqrt(gx * gx + g_perp_sq) # Magnitude of g
        g_perp = np.sqrt(g_perp_sq) # Magnitude of perpendicular component

        # Random azimuthal angle (ϵ) between 0 and 2π for isotropic scattering
        rng = _coerce_rng(rng)
        if rng is None:
            ep = np.random.rand(N) * np.pi * 2.
        else:
            ep = rng.random(N) * np.pi * 2.
        cos_ep: npt.NDArray[float] = np.cos(ep)
        sin_ep: npt.NDArray[float] = np.sin(ep)

        # Compute components of orthogonal vector h (see Nanbu 1997 derivation)
        hx = np.zeros(N)
        hy = np.zeros(N)
        hz = np.zeros(N)
        general = g_perp > 0.0
        axis_aligned = ~general & (g_abs > 0.0)

        if np.any(general):
            gx_cosep = gx[general] * cos_ep[general]
            gabs_sinep = g_abs[general] * sin_ep[general]
            hx[general] = g_perp[general] * cos_ep[general]
            hy[general] = - (gy[general] * gx_cosep + gz[general] * gabs_sinep) / g_perp[general]
            hz[general] = (gy[general] * gabs_sinep - gz[general] * gx_cosep) / g_perp[general]

        if np.any(axis_aligned):
            hy[axis_aligned] = g_abs[axis_aligned] * cos_ep[axis_aligned]
            hz[axis_aligned] = g_abs[axis_aligned] * sin_ep[axis_aligned]

        # Return stacked orthogonal vectors
        return np.stack((hx, hy, hz), axis=1)

    @staticmethod
    def shuffle_rows_with_map(array: np.ndarray, rng: RNGLike = None) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Shuffles the rows of an array and returns the shuffled array with a mapping index.

        :param array: Input array.
        :return: (Shuffled array, row mapping)
        """
        rng = _coerce_rng(rng)
        if rng is None:
            row_map = np.random.permutation(array.shape[0])
        else:
            row_map = rng.permutation(array.shape[0])
        shuffled_array = array[row_map,:]
        return shuffled_array, row_map

    @staticmethod
    def restore_rows_from_map(array: np.ndarray, row_map: np.ndarray) -> npt.NDArray[float]:
        """
        Restore the original row order after `shuffle_rows_with_map`.
        """
        restored = np.empty_like(array)
        restored[row_map] = array
        return restored

    def _get_like_workspace(
        self,
        spc: Particle,
        shuffled_vel: np.ndarray,
    ) -> tuple[int, int, Particle, Particle, "Collision"]:
        key = id(spc)
        N_1 = spc.Nmarker // 2
        N_2 = spc.Nmarker - N_1
        workspace = self._like_workspaces.get(key)
        if workspace is None or workspace[0] != N_1 or workspace[1] != N_2:
            spc_1 = Particle(
                name=f"{spc.name}_1",
                charge=spc.charge / e,
                mass=spc.mass / physical_constants['atomic mass constant'][0],
                density=spc.density,
                weight=spc.weight,
                Nmarker=N_1,
                vel=shuffled_vel[0:N_1],
                rng=self.rng,
            )
            spc_2 = Particle(
                name=f"{spc.name}_2",
                charge=spc.charge / e,
                mass=spc.mass / physical_constants['atomic mass constant'][0],
                density=spc.density,
                weight=spc.weight,
                Nmarker=N_2,
                vel=shuffled_vel[N_1:N_1 + N_2],
                rng=self.rng,
            )
            col_like = Collision(spc_1, spc_2, self.dtp, rng=self.rng)
            workspace = (N_1, N_2, spc_1, spc_2, col_like)
            self._like_workspaces[key] = workspace
        else:
            _, _, spc_1, spc_2, col_like = workspace
            spc_1.assign_vel(shuffled_vel[0:N_1], refresh_stats=True)
            spc_2.assign_vel(shuffled_vel[N_1:N_1 + N_2], refresh_stats=True)
        return workspace

    def _random(self, size) -> npt.NDArray[float]:
        if self.rng is None:
            return np.random.random(size)
        return self.rng.random(size)

    def _permutation(self, size: int) -> npt.NDArray[np.int64]:
        if self.rng is None:
            return np.random.permutation(size)
        return self.rng.permutation(size)

    @classmethod
    def _effective_temperature(cls, species: Particle) -> float:
        temperature = species.temperature_actual
        if temperature is None or not np.isfinite(temperature) or temperature <= 0.0:
            temperature = species.temperature_given
        if temperature is None or not np.isfinite(temperature) or temperature <= 0.0:
            temperature = cls.min_temperature_ev
        return max(float(temperature), cls.min_temperature_ev)
