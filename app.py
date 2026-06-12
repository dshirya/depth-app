"""Laser Penetration Depth Calculator — Streamlit app."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from optics import (
    alpha_from_k,
    k_from_alpha,
    fresnel_R_air,
    fresnel_R_interface,
    penetration_depth,
    d99_simple,
    d99_corrected,
    intensity_profile,
    absorbed_power_density,
    film_absorption_fraction,
    substrate_transmission_fraction,
    tmm_r_t,
    tmm_intensity_profile,
    tmm_absorbed_density,
    tmm_energy_fractions,
    substrate_exit_fraction,
    gaussian_rayleigh_length,
    gaussian_film_params,
    gaussian_substrate_params,
    gaussian_kappa_prime,
    gaussian_omega,
    gaussian_conc_factor,
    FILM_PRESETS,
    SUBSTRATE_PRESETS,
)

st.set_page_config(page_title="Laser Penetration Depth", layout="wide")
st.title("Laser Penetration Depth Calculator")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Laser")
wavelength_nm = st.sidebar.number_input(
    "Wavelength λ [nm]", min_value=100.0, max_value=2000.0, value=355.0, step=1.0
)
I0 = st.sidebar.number_input("Incident intensity I₀", value=1.0, min_value=0.0, step=0.1)
lambda_cm = wavelength_nm * 1e-7

st.sidebar.header("Film")
film_preset = st.sidebar.selectbox("Film preset", list(FILM_PRESETS.keys()))
default_film = FILM_PRESETS[film_preset]

alpha_mode = st.sidebar.toggle("Enter α directly instead of k", value=False)
n_film = st.sidebar.number_input("n (film)", value=float(default_film["n"]), min_value=0.01, step=0.01)

if alpha_mode:
    alpha_film_input = st.sidebar.number_input(
        "α_film [cm⁻¹]",
        value=float(alpha_from_k(default_film["k"], lambda_cm)),
        min_value=0.0, step=1000.0, format="%.3e",
    )
    k_film = k_from_alpha(alpha_film_input, lambda_cm)
    alpha_film = alpha_film_input
else:
    k_film = st.sidebar.number_input("k (film)", value=float(default_film["k"]), min_value=0.0, step=0.01)
    alpha_film = alpha_from_k(k_film, lambda_cm)

d_film_nm = st.sidebar.number_input("Film thickness [nm]", value=200.0, min_value=1.0, step=10.0)
d_film_cm = d_film_nm * 1e-7

st.sidebar.header("Substrate")
sub_preset = st.sidebar.selectbox("Substrate preset", list(SUBSTRATE_PRESETS.keys()))
default_sub = SUBSTRATE_PRESETS[sub_preset]

n_sub = st.sidebar.number_input("n (substrate)", value=float(default_sub["n"]), min_value=0.01, step=0.01)
k_sub = st.sidebar.number_input(
    "k (substrate)", value=float(default_sub["k"]), min_value=0.0, step=0.001, format="%.4f"
)
d_sub_um = st.sidebar.number_input(
    "Substrate thickness [µm]",
    value=1.0, min_value=1.0, max_value=100_000.0, step=1.0,
    help="Physical slab thickness. Controls how far the plot extends and determines how much light exits the back face.",
)
d_sub_cm = d_sub_um * 1e-4

alpha_sub = alpha_from_k(k_sub, lambda_cm)

st.sidebar.header("Calculation method")
use_tmm = st.sidebar.toggle(
    "Include thin-film interference (TMM)",
    value=True,
    help="Transfer Matrix Method accounts for standing-wave interference inside the film. "
         "Important when film thickness ≈ λ/(4n). Disable to use the simpler Beer–Lambert model.",
)

st.sidebar.header("Laser focusing")
omega_0_nm = st.sidebar.number_input(
    "Beam waist ω₀ [nm]",
    value=500.0, min_value=10.0, max_value=1_000_000.0, step=50.0,
    help="1/e² intensity radius at the beam focus. Large values → plane-wave limit.",
)
Delta_nm = st.sidebar.number_input(
    "Focus depth Δ below surface [nm]",
    value=0.0, min_value=-10_000.0, max_value=10_000.0, step=50.0,
    help="Apparent focus position below the air/film interface (measured in the incident medium). "
         "0 = waist at the surface; positive = focusing into the film; negative = focus above surface.",
)
n_inc = st.sidebar.number_input(
    "n of incident medium",
    value=1.0, min_value=0.1, max_value=5.0, step=0.1,
    help="Refractive index of the medium above the film (air = 1.0). Affects Rayleigh length scaling.",
)

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
R_air_fresnel = fresnel_R_air(n_film, k_film)
R_fs = fresnel_R_interface(n_film, k_film, n_sub, k_sub)
R_back = fresnel_R_air(n_sub, k_sub)

delta_film_cm = penetration_depth(alpha_film)
delta_film_nm = delta_film_cm * 1e7 if np.isfinite(delta_film_cm) else np.inf
d99s_cm = d99_simple(alpha_film)
d99s_nm = d99s_cm * 1e7 if np.isfinite(d99s_cm) else np.inf
d99c_cm = d99_corrected(alpha_film, R_air_fresnel)
d99c_nm = d99c_cm * 1e7 if np.isfinite(d99c_cm) else np.inf

delta_sub_cm = penetration_depth(alpha_sub)
delta_sub_nm = delta_sub_cm * 1e7 if (k_sub > 0 and np.isfinite(delta_sub_cm)) else None

if use_tmm:
    R_total, A_film_frac, T_total = tmm_energy_fractions(
        I0, lambda_cm, n_film, k_film, d_film_cm, n_sub, k_sub
    )
    R_display = R_total
    A_film_display = A_film_frac
    T_display = T_total
else:
    R_display = R_air_fresnel
    A_film_display = film_absorption_fraction(I0, R_air_fresnel, alpha_film, d_film_cm, R_fs) / I0
    T_display = substrate_transmission_fraction(I0, R_air_fresnel, alpha_film, d_film_cm, R_fs) / I0

I_exit_frac = substrate_exit_fraction(I0, T_display, alpha_sub, d_sub_cm, n_sub, k_sub) / I0

# Gaussian beam physics
z_R_inc_nm = gaussian_rayleigh_length(n_inc, omega_0_nm, wavelength_nm)
z0_c_nm, zw_c_nm = gaussian_film_params(n_inc, n_film, z_R_inc_nm, Delta_nm)
z0_s_nm, zw_s_nm = gaussian_substrate_params(n_inc, n_film, n_sub, z_R_inc_nm, Delta_nm, d_film_nm)
kappa_prime_c = gaussian_kappa_prime(n_film, k_film)
kappa_prime_s = gaussian_kappa_prime(n_sub, k_sub)
validity_ratio = d_film_nm / z0_c_nm if z0_c_nm > 0 else 0.0

# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------
if np.isfinite(delta_film_nm) and d_film_nm < delta_film_nm:
    st.warning(
        f"Film thickness ({d_film_nm:.0f} nm) is **less than the 1/e depth δ = {delta_film_nm:.0f} nm**. "
        "The laser reaches the substrate — substrate PL may contribute to measured spectra."
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
st.subheader("Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Surface reflectance",
    f"{R_display * 100:.1f}%",
    help="Fraction of incident light reflected at the top surface. "
         "TMM value includes interference from all interfaces; "
         "simple Fresnel: R = ((n−1)²+k²) / ((n+1)²+k²).",
)
c2.metric(
    "Interface reflectance (film/sub)",
    f"{R_fs * 100:.1f}%",
    help="Single-interface Fresnel step at the film/substrate boundary: "
         "R_fs = ((n_f−n_s)²+(k_f−k_s)²) / ((n_f+n_s)²+(k_f+k_s)²).",
)
c3.metric(
    "Absorption coefficient α",
    f"{alpha_film:.3e} cm⁻¹",
    help="α = 4πk/λ [cm⁻¹]. Controls how quickly intensity decays inside the film.",
)
c4.metric(
    "k_film" + (" (derived)" if alpha_mode else ""),
    f"{k_film:.4f}",
    help="Extinction coefficient — imaginary part of the complex refractive index ñ = n + ik.",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric(
    "1/e penetration depth δ",
    f"{delta_film_nm:.0f} nm" if np.isfinite(delta_film_nm) else "∞",
    help="Depth at which intensity falls to 1/e ≈ 37% of the value just inside the film. "
         "δ = 1/α = λ/(4πk). Independent of surface reflectance.",
)
c6.metric(
    "99% absorption depth",
    f"{d99s_nm:.0f} nm" if np.isfinite(d99s_nm) else "∞",
    help="Depth at which 99% of the transmitted (post-reflection) light is absorbed: "
         "d₉₉ = ln(100)/α.",
)
c7.metric(
    "99% total attenuation depth",
    f"{d99c_nm:.0f} nm" if np.isfinite(d99c_nm) else "∞",
    help="Depth at which only 1% of the incident light remains (reflection + absorption combined): "
         "d₉₉ = ln(100·(1−R))/α.",
)
if delta_sub_nm is not None:
    c8.metric(
        "1/e depth in substrate δ_sub",
        f"{delta_sub_nm:.0f} nm",
        help="Beer–Lambert 1/e penetration depth inside the substrate.",
    )
else:
    c8.metric(
        "1/e depth in substrate δ_sub",
        "∞ (transparent)",
        help="k_sub = 0 → no absorption in substrate.",
    )

c9, c10, c11 = st.columns(3)
c9.metric(
    "Light absorbed in film",
    f"{A_film_display * 100:.1f}%",
    help="Fraction of incident I₀ absorbed before reaching the substrate.",
)
c10.metric(
    "Light entering substrate",
    f"{T_display * 100:.1f}%",
    help="Fraction of I₀ transmitted through the film and into the substrate.",
)
c11.metric(
    "Light exiting back face",
    f"{I_exit_frac * 100:.2f}%",
    help=f"Fraction surviving Beer–Lambert decay across {d_sub_um:.0f} µm substrate, "
         "minus Fresnel reflection at the substrate/air back face.",
)

R_pct = R_display * 100
A_pct = A_film_display * 100
T_pct = T_display * 100
st.caption(
    f"Energy check: reflected {R_pct:.1f}% + absorbed in film {A_pct:.1f}% + "
    f"into substrate {T_pct:.1f}% = {R_pct+A_pct+T_pct:.1f}%"
)

with st.expander("Physics reference — formulas"):
    fc1, fc2 = st.columns(2)
    with fc1:
        st.write("**Absorption coefficient**")
        st.latex(r"\alpha = \frac{4\pi k}{\lambda}")
        st.write("**Beer–Lambert intensity decay**")
        st.latex(r"I(z) = I_0\,(1-R)\,e^{-\alpha z}")
        st.write("**1/e penetration depth**")
        st.latex(r"\delta = \frac{1}{\alpha} = \frac{\lambda}{4\pi k}")
        st.write("**99% attenuation depth (reflection-corrected)**")
        st.latex(r"d_{99} = \frac{\ln\!\bigl(100\,(1-R)\bigr)}{\alpha}")
        st.write("**Gaussian Rayleigh length**")
        st.latex(r"z_R = \frac{\pi\,n_{\rm inc}\,\omega_0^2}{\lambda}")
        st.write("**Waist position after refraction into film**")
        st.latex(r"z_w = \frac{n_{\rm film}}{n_{\rm inc}}\,\Delta")
    with fc2:
        st.write("**Air/film Fresnel reflectance**")
        st.latex(r"R = \frac{(n-1)^2 + k^2}{(n+1)^2 + k^2}")
        st.write("**Film/substrate interface**")
        st.latex(
            r"R_{fs} = \frac{(n_f - n_s)^2 + (k_f - k_s)^2}"
            r"{(n_f + n_s)^2 + (k_f + k_s)^2}"
        )
        st.write("**TMM transfer matrix (Born & Wolf)**")
        st.latex(
            r"M = \begin{pmatrix}\cos\phi & -\tfrac{i}{N_1}\sin\phi \\"
            r"-iN_1\sin\phi & \cos\phi\end{pmatrix}"
            r"\quad \phi=\tfrac{2\pi N_1 d}{\lambda}"
        )
        st.write("**Beam radius in absorbing medium**")
        st.latex(
            r"\omega(z)=\omega_0\sqrt{\frac{\zeta^2+1+2\kappa'\zeta}{\kappa'\zeta+1}},"
            r"\quad\zeta=\frac{z-z_w}{z_0},\quad\kappa'=\frac{\kappa}{\sqrt{n^2+\kappa^2}}"
        )
        st.write("**Mode B generation rate (Gaussian-corrected)**")
        st.latex(r"G_B(z)=\alpha\,I_{\rm TMM}(z)\!\left(\frac{\omega_0}{\omega(z)}\right)^{\!2}")

    with st.expander("Derivation note: why d₉₉ depends on R but δ does not"):
        st.markdown(
            "The **(1−R)** prefactor cancels when solving for the 1/e depth δ, "
            "so **δ is independent of R**. It does NOT cancel for d₉₉ when defined "
            "relative to the incident intensity:\n\n"
            "$(1-R)\\,e^{-\\alpha z} = 0.01$\n\n"
            "gives $z = \\ln(100\\cdot(1-R))/\\alpha$, which is slightly **shallower** "
            "than $\\ln(100)/\\alpha$."
        )

with st.expander("How to measure k for your own films"):
    st.markdown(
        "**Spectroscopic ellipsometry** is the gold standard: it measures the complex "
        "ratio of p- and s-polarized reflectances and fits n and k simultaneously.\n\n"
        "A quick UV-Vis estimate uses transmittance T and reflectance R on a film of known thickness d:"
    )
    st.latex(r"\alpha = \frac{1}{d}\,\ln\!\left(\frac{(1-R)^2}{T}\right)")
    st.markdown("Convert to k via $k = \\alpha\\lambda/(4\\pi)$.")

# ---------------------------------------------------------------------------
# Gaussian beam results
# ---------------------------------------------------------------------------
st.subheader("Gaussian beam parameters")

gc1, gc2, gc3, gc4 = st.columns(4)
gc1.metric(
    "Rayleigh length z_R (air)",
    f"{z_R_inc_nm:.0f} nm",
    help="z_R = π·n_inc·ω₀²/λ. Distance from the waist at which beam area doubles.",
)
gc2.metric(
    "Rayleigh length in film z₀",
    f"{z0_c_nm:.0f} nm",
    help="z₀ = (n_film/n_inc)·z_R. Scales with the film refractive index after refraction.",
)
gc3.metric(
    "Waist position in film z_w",
    f"{zw_c_nm:.0f} nm",
    help="z_w = (n_film/n_inc)·Δ. Geometric waist shifts deeper due to refraction.",
)
gc4.metric(
    "d_film / z₀ (validity ratio)",
    f"{validity_ratio:.2f}",
    help="< 0.2: plane-wave model accurate. 0.2–1.0: moderate focusing. > 1.0: correction important.",
)

if validity_ratio < 0.2:
    st.success(
        f"Validity ratio {validity_ratio:.2f} < 0.2 — Gaussian correction negligible. "
        "The plane-wave model is accurate for this focus geometry."
    )
elif validity_ratio < 1.0:
    st.info(
        f"Validity ratio {validity_ratio:.2f} (0.2–1.0) — Moderate beam focusing. "
        "Gaussian correction reshapes the generation profile noticeably near the waist."
    )
else:
    st.warning(
        f"Validity ratio {validity_ratio:.2f} > 1.0 — Tight focus. "
        "Gaussian beam correction significantly alters the excitation depth profile."
    )

with st.expander("Why the plane-wave model is not enough (focused beams)"):
    st.markdown("""
The Beer–Lambert / TMM model treats the laser as a plane wave: uniform transverse profile and
no axial variation except from absorption. This breaks down when the beam is tightly focused:

**a. Axial intensity gradient near the waist.** A Gaussian beam has a finite Rayleigh length
z₀ inside the film. The beam area ω(z)² varies along the optical axis — smallest at the waist,
growing on either side. The on-axis intensity (∝ 1/ω²) therefore peaks at the waist, adding an
axial modulation on top of Beer–Lambert absorption.

**b. The carrier generation profile is compressed near the waist.** Even if every absorbed photon
has the same probability of generating a carrier, more carriers are created per nm³ at the waist
because more photons pass through each unit volume there. Mode B corrects for this by weighting
G(z) = α · I(z) · (ω₀/ω)².

**c. Waist position shifts upon refraction.** When the beam crosses the air/film interface, the
paraxial Snell condition maps the apparent focus depth Δ to z_w = (n_film/n_inc)·Δ inside the
film. For n_film/n_inc = 2, the waist is twice as deep as expected from the geometric air-side
setting — ignoring this gives a wrong estimate of the excitation centroid.

**d. Validity criterion.** The correction is significant when d_film / z₀ > 0.2. If the film is
much thinner than the Rayleigh length, the beam radius barely changes across the film and the
plane-wave model is fine. See validity ratio above.
""")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
st.subheader("Depth profiles")

plot_col, ctrl_col = st.columns([5, 1])
with ctrl_col:
    log_scale = st.toggle("Log y-axis", value=False)
    show_gen = st.toggle("Show absorption profile", value=True)
    if show_gen:
        gauss_mode = st.radio(
            "Mode",
            ["A (plane-wave)", "B (Gaussian)"],
            index=0,
            help="Mode A: standard TMM/Beer–Lambert generation profile. "
                 "Mode B: Gaussian-corrected — weighted by (ω₀/ω)², normalized to same total energy.",
        )
    else:
        gauss_mode = "A (plane-wave)"

# Depth arrays
z_film_nm = np.linspace(0, d_film_nm, 500)
z_sub_nm = np.linspace(0, d_sub_um * 1e3, 500)
z_film_cm = z_film_nm * 1e-7
z_sub_cm = z_sub_nm * 1e-7

if use_tmm:
    I_film, I_sub = tmm_intensity_profile(
        z_film_cm, z_sub_cm, lambda_cm, n_film, k_film, d_film_cm, n_sub, k_sub, I0
    )
    gen_film, gen_sub = tmm_absorbed_density(
        z_film_cm, z_sub_cm, lambda_cm, n_film, k_film, d_film_cm, n_sub, k_sub, I0
    )
else:
    I_film, I_sub = intensity_profile(
        z_film_cm, z_sub_cm, I0, R_air_fresnel, alpha_film, d_film_cm, R_fs, alpha_sub
    )
    gen_film, gen_sub = absorbed_power_density(
        z_film_cm, z_sub_cm, I0, R_air_fresnel, alpha_film, d_film_cm, R_fs, alpha_sub
    )

# Gaussian concentration factors — substrate z is measured from film/substrate interface
cf_film = gaussian_conc_factor(z_film_nm, omega_0_nm, zw_c_nm, z0_c_nm, kappa_prime_c)
cf_sub = gaussian_conc_factor(z_sub_nm, omega_0_nm, zw_s_nm, z0_s_nm, kappa_prime_s)

# Mode A and Mode B generation profiles in I₀/nm
gen_film_A = gen_film * 1e-7
gen_sub_A = gen_sub * 1e-7
gen_film_B_raw = gen_film * cf_film * 1e-7
gen_sub_B_raw = gen_sub * cf_sub * 1e-7

# Normalize B to same total absorbed energy as A
sum_A = np.trapezoid(gen_film_A, z_film_nm) + np.trapezoid(gen_sub_A, z_sub_nm)
sum_B = np.trapezoid(gen_film_B_raw, z_film_nm) + np.trapezoid(gen_sub_B_raw, z_sub_nm)
norm_B = (sum_A / sum_B) if (sum_B > 0 and sum_A > 0) else 1.0
gen_film_B = gen_film_B_raw * norm_B
gen_sub_B = gen_sub_B_raw * norm_B

z_backface_nm = d_film_nm + d_sub_um * 1e3
I_sub_back = I_sub[-1] * (1 - R_back)

n_plots = 2 if show_gen else 1
fig, axes = plt.subplots(1, n_plots, figsize=(13 if show_gen else 7, 5), sharey=False)
if n_plots == 1:
    axes = [axes]

ax1 = axes[0]
ax1.axvspan(d_film_nm, z_backface_nm, alpha=0.10, color="steelblue")
ax1.plot(z_film_nm, I_film / I0, color="crimson", lw=2, label="Film")
ax1.plot(d_film_nm + z_sub_nm, I_sub / I0, color="steelblue", lw=2, label="Substrate")
ax1.axvline(z_backface_nm, color="navy", lw=1.5, ls="--", label=f"Back face ({d_sub_um:.0f} µm)")
ax1.plot([z_backface_nm, z_backface_nm], [I_sub[-1] / I0, I_sub_back / I0], color="navy", lw=2)
ax1.axvline(d_film_nm, color="black", lw=1.5, ls="--", label=f"Interface ({d_film_nm:.0f} nm)")

if np.isfinite(delta_film_nm) and delta_film_nm < d_film_nm:
    yval_delta = (1 - R_display) * np.exp(-1)
    ax1.axvline(delta_film_nm, color="orange", lw=1, ls=":", alpha=0.9)
    ax1.annotate(
        f"δ = {delta_film_nm:.0f} nm",
        xy=(delta_film_nm, yval_delta / I0),
        xytext=(delta_film_nm + d_film_nm * 0.07, yval_delta / I0 * 1.15),
        fontsize=10, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8),
    )

if np.isfinite(d99c_nm) and d99c_nm < z_backface_nm:
    ax1.axvline(d99c_nm, color="purple", lw=1, ls=":", alpha=0.9)
    ax1.annotate(
        f"d₉₉ = {d99c_nm:.0f} nm",
        xy=(d99c_nm, 0.01),
        xytext=(d99c_nm + d_film_nm * 0.07, 0.04),
        fontsize=10, color="purple",
        arrowprops=dict(arrowstyle="->", color="purple", lw=0.8),
    )

ax1.set_xlabel("Depth z [nm]", fontsize=12)
ax1.set_ylabel("I(z) / I₀", fontsize=12)
ax1.set_title("Laser intensity vs depth", fontsize=12)
if log_scale:
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-4)
else:
    ax1.set_ylim(bottom=0)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

if show_gen:
    ax2 = axes[1]
    use_mode_B = (gauss_mode == "B (Gaussian)")

    ax2.axvspan(d_film_nm, z_backface_nm, alpha=0.10, color="steelblue")
    ax2.axvline(d_film_nm, color="black", lw=1.5, ls="--")
    ax2.axvline(z_backface_nm, color="navy", lw=1.5, ls="--")

    # Mode A — gray dashed; dimmed when Mode B is active
    alpha_A = 0.45 if use_mode_B else 1.0
    ax2.plot(z_film_nm, gen_film_A, color="gray", lw=1.5, ls="--",
             alpha=alpha_A, label="Mode A (plane-wave)")
    ax2.plot(d_film_nm + z_sub_nm, gen_sub_A, color="gray", lw=1.5, ls="--", alpha=alpha_A)
    if not use_mode_B:
        ax2.fill_between(z_film_nm, gen_film_A, alpha=0.25, color="crimson")
        ax2.fill_between(d_film_nm + z_sub_nm, gen_sub_A, alpha=0.25, color="steelblue")

    # Mode B — solid colored; dimmed when Mode A is active
    alpha_B = 1.0 if use_mode_B else 0.45
    ax2.plot(z_film_nm, gen_film_B, color="crimson", lw=2, alpha=alpha_B, label="Mode B (Gaussian)")
    ax2.plot(d_film_nm + z_sub_nm, gen_sub_B, color="steelblue", lw=2, alpha=alpha_B)
    if use_mode_B:
        ax2.fill_between(z_film_nm, gen_film_B, alpha=0.25, color="crimson")
        ax2.fill_between(d_film_nm + z_sub_nm, gen_sub_B, alpha=0.25, color="steelblue")

    ax2.set_xlabel("Depth z [nm]", fontsize=12)
    ax2.set_ylabel("Absorbed intensity [I₀/nm]", fontsize=12)
    ax2.set_title("Where is the laser absorbed?\n(PL excitation depth profile)", fontsize=12)
    if log_scale:
        ax2.set_yscale("log")
        peak = max(
            gen_film_A.max() if gen_film_A.size else 1e-20,
            gen_film_B.max() if gen_film_B.size else 1e-20,
            gen_sub_A.max() if gen_sub_A.size else 1e-20,
            gen_sub_B.max() if gen_sub_B.size else 1e-20,
        )
        ax2.set_ylim(bottom=max(peak * 1e-5, 1e-20))
    else:
        ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
with plot_col:
    st.pyplot(fig)
plt.close(fig)

if show_gen:
    st.info(
        "**Reading the absorption profile:** Each point shows laser energy deposited per nm at that depth. "
        "Area under the **film curve** (pink) vs **substrate curve** (blue) is proportional to the "
        "fraction of PL signal from each layer. "
        "**Mode A** (gray dashed) = plane-wave. **Mode B** (solid) = Gaussian-corrected, "
        "normalized to the same total absorbed energy."
    )

if use_tmm:
    st.caption(
        "Oscillations inside the film region are **thin-film interference** (standing wave). "
        "Most prominent when film thickness ≈ λ/(4n)."
    )

# ---------------------------------------------------------------------------
# Gaussian beam envelope plot
# ---------------------------------------------------------------------------
st.subheader("Gaussian beam envelope")

# Air: from -2·z_R to the surface
z_air_plot = np.linspace(-2.0 * z_R_inc_nm, 0.0, 400)
omega_air_plot = omega_0_nm * np.sqrt(1.0 + ((z_air_plot - Delta_nm) / z_R_inc_nm) ** 2)

# Film: 0 to d_film_nm (same array as depth profiles)
omega_film_plot = gaussian_omega(z_film_nm, omega_0_nm, zw_c_nm, z0_c_nm, kappa_prime_c)

# Substrate: show up to 3·z0_s or full substrate, whichever is shorter
z_sub_env_max = min(d_sub_um * 1e3, 3.0 * z0_s_nm + 200.0)
z_sub_env = np.linspace(0.0, z_sub_env_max, 400)
omega_sub_plot = gaussian_omega(z_sub_env, omega_0_nm, zw_s_nm, z0_s_nm, kappa_prime_s)

z_env = np.concatenate([z_air_plot, z_film_nm, d_film_nm + z_sub_env])
omega_env = np.concatenate([omega_air_plot, omega_film_plot, omega_sub_plot])

fig2, ax_env = plt.subplots(figsize=(12, 4))

ax_env.axvspan(z_air_plot[0], 0.0, alpha=0.06, color="gray")
ax_env.axvspan(0.0, d_film_nm, alpha=0.12, color="crimson")
ax_env.axvspan(d_film_nm, d_film_nm + z_sub_env_max, alpha=0.08, color="steelblue")

ax_env.fill_between(z_env, -omega_env, omega_env, alpha=0.25, color="gold")
ax_env.plot(z_env, omega_env, color="darkorange", lw=2, label="ω(z)")
ax_env.plot(z_env, -omega_env, color="darkorange", lw=2)

ax_env.axhline(omega_0_nm, color="dimgray", ls=":", lw=1, label=f"ω₀ = {omega_0_nm:.0f} nm")
ax_env.axhline(-omega_0_nm, color="dimgray", ls=":", lw=1)

ax_env.axvline(0.0, color="black", lw=1.5, ls="--")
ax_env.axvline(d_film_nm, color="black", lw=1.5, ls="--")

# Label regions
y_top = omega_env.max() * 0.88
ax_env.text(-z_R_inc_nm, y_top, "Air", fontsize=10, color="gray", ha="center")
ax_env.text(d_film_nm / 2, y_top, "Film", fontsize=10, color="crimson", ha="center")
ax_env.text(d_film_nm + z_sub_env_max / 2, y_top, "Substrate",
            fontsize=10, color="steelblue", ha="center")

# Waist annotation (only if inside the film)
if 0.0 <= zw_c_nm <= d_film_nm:
    ax_env.axvline(zw_c_nm, color="crimson", lw=1, ls=":", alpha=0.8)
    ax_env.annotate(
        f"film waist\nz_w = {zw_c_nm:.0f} nm",
        xy=(zw_c_nm, omega_0_nm * 0.05),
        xytext=(zw_c_nm + d_film_nm * 0.3, omega_env.max() * 0.35),
        fontsize=10, color="crimson", ha="left",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
    )

ax_env.set_xlabel("Depth z [nm]  (negative = air above film)")
ax_env.set_ylabel("Beam radius ω(z) [nm]")
ax_env.set_title("Gaussian beam envelope through the film stack")
ax_env.legend(fontsize=10, loc="lower right")
ax_env.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "The gold band shows ±ω(z), the 1/e² intensity radius along the optical axis. "
    "Absorption damping in the film is accounted for via κ' = κ/√(n²+κ²). "
    "Paraxial, on-axis approximation — valid when ω₀ ≫ λ."
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "**v1 limitations:** Normal incidence only. TMM covers coherent single-film interference "
    "but excludes full multilayer stacks, incoherent multiple reflections at the back face, "
    "and non-normal incidence. Gaussian correction uses the paraxial on-axis approximation "
    "(valid when ω₀ ≫ λ). See README.md for details."
)
