"""pytest validation tests against known literature values."""

import numpy as np
import pytest
from optics import (
    alpha_from_k,
    k_from_alpha,
    fresnel_R_air,
    fresnel_R_interface,
    penetration_depth,
    d99_simple,
    d99_corrected,
    film_absorption_fraction,
    substrate_transmission_fraction,
    tmm_r_t,
    tmm_intensity_profile,
    tmm_energy_fractions,
    gaussian_rayleigh_length,
    gaussian_film_params,
    gaussian_substrate_params,
    gaussian_kappa_prime,
    gaussian_omega,
    gaussian_conc_factor,
)

LAMBDA_355_CM = 355e-7  # 355 nm in cm


def test_alpha_zno_355nm():
    """ZnO k=0.55 at 355 nm -> alpha ≈ 1.95e5 cm^-1."""
    k = 0.55
    alpha = alpha_from_k(k, LAMBDA_355_CM)
    assert abs(alpha - 1.95e5) / 1.95e5 < 0.01, f"alpha={alpha:.3e}"


def test_delta_zno_355nm():
    """ZnO k=0.55 at 355 nm -> delta ≈ 51 nm (±1 nm)."""
    k = 0.55
    alpha = alpha_from_k(k, LAMBDA_355_CM)
    delta_nm = penetration_depth(alpha) * 1e7  # cm -> nm
    assert abs(delta_nm - 51) <= 1, f"delta={delta_nm:.1f} nm"


def test_fresnel_R_zno():
    """R for n=2.0, k=0.55 should be ≈ 13-14% (CLAUDE.md says ~13%, formula gives 14.0%)."""
    R = fresnel_R_air(2.0, 0.55)
    assert 0.12 < R < 0.16, f"R={R:.4f}"


def test_delta_formula():
    """delta [nm] = lambda [nm] / (4 * pi * k)."""
    k = 0.55
    lam_nm = 355
    delta_expected = lam_nm / (4 * np.pi * k)
    alpha = alpha_from_k(k, lam_nm * 1e-7)
    delta_calc = penetration_depth(alpha) * 1e7
    assert abs(delta_calc - delta_expected) < 0.01, f"delta={delta_calc:.2f} vs {delta_expected:.2f}"


def test_k_round_trip():
    """k -> alpha -> k round-trip."""
    k_orig = 0.55
    alpha = alpha_from_k(k_orig, LAMBDA_355_CM)
    k_back = k_from_alpha(alpha, LAMBDA_355_CM)
    assert abs(k_back - k_orig) < 1e-10


def test_d99_simple_greater_than_delta():
    """d99_simple must be > delta for any alpha > 0."""
    alpha = 1.95e5
    assert d99_simple(alpha) > penetration_depth(alpha)


def test_d99_corrected_less_than_simple():
    """Reflection-corrected d99 must be <= d99_simple (R >= 0)."""
    alpha = 1.95e5
    R = fresnel_R_air(2.1, 0.55)
    assert d99_corrected(alpha, R) <= d99_simple(alpha)


def test_fresnel_interface_symmetric_loss():
    """R_fs(n1,k1,n2,k2) == R_fs(n2,k2,n1,k1) — formula is symmetric."""
    R1 = fresnel_R_interface(2.1, 0.55, 1.48, 0.0)
    R2 = fresnel_R_interface(1.48, 0.0, 2.1, 0.55)
    assert abs(R1 - R2) < 1e-10


def test_zno_film_absorption():
    """
    ZnO 200 nm on fused silica at 355 nm.
    The fraction absorbed in the film should exceed 97% of the transmitted light.
    """
    k = 0.55
    n_f, n_s = 2.10, 1.48
    alpha_film = alpha_from_k(k, LAMBDA_355_CM)
    R_air = fresnel_R_air(n_f, k)
    R_fs = fresnel_R_interface(n_f, k, n_s, 0.0)
    d_film_cm = 200e-7
    A = film_absorption_fraction(1.0, R_air, alpha_film, d_film_cm, R_fs)
    T_sub = substrate_transmission_fraction(1.0, R_air, alpha_film, d_film_cm, R_fs)
    # Fraction absorbed relative to what entered the film
    fraction_of_transmitted = A / (1 - R_air)
    assert fraction_of_transmitted > 0.97, f"absorbed fraction of transmitted = {fraction_of_transmitted:.4f}"


def test_transparent_substrate_flat():
    """k_sub=0 means no decay in substrate — substrate intensity should be constant."""
    from optics import intensity_profile
    z_sub = np.linspace(0, 1000e-7, 200)  # 0-1000 nm
    _, I_sub = intensity_profile(
        z_film=np.array([0.0]),
        z_sub=z_sub,
        I0=1.0,
        R_air=0.13,
        alpha_film=1.95e5,
        d_film_cm=200e-7,
        R_fs=0.01,
        alpha_sub=0.0,
    )
    # All substrate values should be equal (flat)
    assert np.allclose(I_sub, I_sub[0]), "Non-flat substrate for k_sub=0"


# ---------------------------------------------------------------------------
# TMM tests
# ---------------------------------------------------------------------------

def test_tmm_R_T_bounded():
    """TMM R and T must be in [0, 1] and R+T <= 1."""
    R, T, r, t = tmm_r_t(LAMBDA_355_CM, 2.10, 0.55, 200e-7, 1.48, 0.0)
    assert 0 <= R <= 1, f"R={R}"
    assert 0 <= T <= 1, f"T={T}"
    assert R + T <= 1 + 1e-9, f"R+T={R+T}"


def test_tmm_energy_conservation():
    """R_TMM + A_film_TMM + T_TMM = 1 (energy conservation)."""
    R, A, T = tmm_energy_fractions(1.0, LAMBDA_355_CM, 2.10, 0.55, 200e-7, 1.48, 0.0)
    assert abs(R + A + T - 1.0) < 1e-4, f"R+A+T = {R+A+T}"


def test_tmm_reduces_to_fresnel_for_zero_thickness():
    """TMM with d=0 should give the same R as single Fresnel air/substrate interface."""
    n_s, k_s = 1.48, 0.0
    R_tmm, T_tmm, _, _ = tmm_r_t(LAMBDA_355_CM, 1.0, 0.0, 0.0, n_s, k_s)
    R_fresnel = fresnel_R_air(n_s, k_s)
    assert abs(R_tmm - R_fresnel) < 1e-6, f"TMM R={R_tmm:.6f} vs Fresnel R={R_fresnel:.6f}"


def test_tmm_interference_fringes_visible():
    """For ZnO 80 nm film (d ~ lambda/4n), TMM intensity must NOT be purely exponential."""
    z_film = np.linspace(0, 80e-7, 300)
    z_sub = np.linspace(0, 200e-7, 100)
    I_film, _ = tmm_intensity_profile(
        z_film, z_sub, LAMBDA_355_CM, 2.10, 0.55, 80e-7, 1.48, 0.0
    )
    # If purely exponential, log(I) would be perfectly linear.
    # Check residual from linear fit > threshold (fringes present).
    log_I = np.log(np.clip(I_film, 1e-12, None))
    z_norm = np.linspace(0, 1, len(log_I))
    coeffs = np.polyfit(z_norm, log_I, 1)
    residual_rms = np.sqrt(np.mean((log_I - np.polyval(coeffs, z_norm)) ** 2))
    assert residual_rms > 0.005, f"No interference fringes detected (residual_rms={residual_rms:.5f})"


# ---------------------------------------------------------------------------
# Gaussian beam tests
# ---------------------------------------------------------------------------

def test_gaussian_kappa0_omega():
    """kappa=0: omega(z) must equal omega_0*sqrt(1+zeta^2) to rtol 1e-10."""
    omega_0, z0, zw = 500.0, 2000.0, 0.0
    z = np.linspace(0, 5000, 200)
    omega_calc = gaussian_omega(z, omega_0, zw, z0, kappa_prime=0.0)
    zeta = (z - zw) / z0
    omega_expected = omega_0 * np.sqrt(1.0 + zeta ** 2)
    np.testing.assert_allclose(omega_calc, omega_expected, rtol=1e-10)


def test_gaussian_planewave_limit():
    """omega_0=1e9 nm: conc_factor ≈ 1 everywhere in film (rtol 1e-4)."""
    omega_0_huge = 1e9
    lam_nm = 355.0
    n_inc, n_c, k_c = 1.0, 2.10, 0.55
    z_R = gaussian_rayleigh_length(n_inc, omega_0_huge, lam_nm)
    z0_c, zw_c = gaussian_film_params(n_inc, n_c, z_R, 0.0)
    kp = gaussian_kappa_prime(n_c, k_c)
    z = np.linspace(0, 200, 100)
    cf = gaussian_conc_factor(z, omega_0_huge, zw_c, z0_c, kp)
    np.testing.assert_allclose(cf, 1.0, rtol=1e-4)


def test_gaussian_waist_shift():
    """n_inc=1, n_c=2.0, Delta=100 nm → zw_c = 200 nm."""
    z_R = gaussian_rayleigh_length(1.0, 500.0, 355.0)
    _, zw_c = gaussian_film_params(1.0, 2.0, z_R, 100.0)
    assert abs(zw_c - 200.0) < 1e-10, f"zw_c = {zw_c}"


def test_gaussian_substrate_waist():
    """n_inc=1, n_c=2.0, n_s=1.5, Delta=100, d_film=300 → zw_s = 225 nm."""
    z_R = gaussian_rayleigh_length(1.0, 500.0, 355.0)
    _, zw_s = gaussian_substrate_params(1.0, 2.0, 1.5, z_R, 100.0, 300.0)
    assert abs(zw_s - 225.0) < 1e-10, f"zw_s = {zw_s}"


def test_gaussian_rayleigh_scaling():
    """z0_c = (n_c/n_inc) * z_R_inc exactly."""
    n_inc, n_c = 1.0, 2.0
    z_R = gaussian_rayleigh_length(n_inc, 500.0, 355.0)
    z0_c, _ = gaussian_film_params(n_inc, n_c, z_R, 0.0)
    assert abs(z0_c - (n_c / n_inc) * z_R) < 1e-10


def test_gaussian_mode_B_larger_near_waist():
    """Normalised mode B exceeds mode A near the focus (Gaussian concentrates excitation)."""
    # Tight focus with waist clearly inside the film
    omega_0, lam_nm = 50.0, 355.0
    n_inc, n_c, k_c = 1.0, 2.10, 0.55
    Delta = 50.0   # focus 50 nm below surface
    d_film = 200.0

    z_R = gaussian_rayleigh_length(n_inc, omega_0, lam_nm)
    z0_c, zw_c = gaussian_film_params(n_inc, n_c, z_R, Delta)
    kp = gaussian_kappa_prime(n_c, k_c)

    z = np.linspace(0, d_film, 400)
    alpha = alpha_from_k(k_c, lam_nm * 1e-7)
    I_ref = np.ones_like(z)   # simplified flat reference intensity

    G_A = alpha * I_ref
    G_B = alpha * I_ref * gaussian_conc_factor(z, omega_0, zw_c, z0_c, kp)

    # Normalise B to same total absorbed energy as A
    G_B_norm = G_B * (G_A.sum() / G_B.sum())

    # At the waist, normalised mode B must exceed mode A
    iz = int(np.argmin(np.abs(z - zw_c)))
    assert G_B_norm[iz] > G_A[iz], (
        f"Mode B not larger at waist: B_norm={G_B_norm[iz]:.4f} vs A={G_A[iz]:.4f}"
    )
