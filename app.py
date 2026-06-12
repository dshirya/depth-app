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
    value=1.0, min_value=1.0, max_value=100_000.0, step=100.0,
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

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
# Simple Fresnel (always computed for reference / Beer–Lambert fallback)
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

# Back-face exit fraction
I_exit_frac = substrate_exit_fraction(I0, T_display, alpha_sub, d_sub_cm, n_sub, k_sub) / I0

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
    help="α = 4πk/λ [cm⁻¹]. Controls how quickly intensity decays inside the film. "
         "A larger α means more of the laser is absorbed near the surface.",
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
         "d₉₉ = ln(100)/α. Does not account for surface reflection losses.",
)
c7.metric(
    "99% total attenuation depth",
    f"{d99c_nm:.0f} nm" if np.isfinite(d99c_nm) else "∞",
    help="Depth at which only 1% of the incident light remains (reflection + absorption combined): "
         "d₉₉ = ln(100·(1−R))/α. Slightly shallower than the simple variant because some light "
         "is already reflected at the surface.",
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
        help="k_sub = 0 → no absorption in substrate (flat intensity profile).",
    )

c9, c10, c11 = st.columns(3)
c9.metric(
    "Light absorbed in film",
    f"{A_film_display * 100:.1f}%",
    help="Fraction of incident I₀ absorbed before reaching the substrate. "
         "With TMM: 1 − R_total − T_total. Without TMM: Beer–Lambert estimate.",
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
         "minus Fresnel reflection at the substrate/air back face. "
         "R_back = ((n_sub−1)²+k_sub²) / ((n_sub+1)²+k_sub²).",
)

# Energy balance line
R_pct = R_display * 100
A_pct = A_film_display * 100
T_pct = T_display * 100
st.caption(
    f"Energy check: reflected {R_pct:.1f}% + absorbed in film {A_pct:.1f}% + "
    f"into substrate {T_pct:.1f}% = {R_pct+A_pct+T_pct:.1f}%"
)

# Reference expanders
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
# Plots
# ---------------------------------------------------------------------------
st.subheader("Depth profiles")

plot_col, ctrl_col = st.columns([5, 1])
with ctrl_col:
    log_scale = st.toggle("Log y-axis", value=False)
    show_gen = st.toggle("Show absorption profile", value=True)

# Depth arrays
z_film_nm = np.linspace(0, d_film_nm, 500)
z_sub_nm = np.linspace(0, d_sub_um * 1e3, 500)  # µm → nm for plot axis
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

# Back-face step for the intensity plot
z_backface_nm = d_film_nm + d_sub_um * 1e3
I_sub_back = I_sub[-1] * (1 - R_back)

n_plots = 2 if show_gen else 1
fig, axes = plt.subplots(1, n_plots, figsize=(13 if show_gen else 7, 5), sharey=False)
if n_plots == 1:
    axes = [axes]

ax1 = axes[0]

# Substrate shading
ax1.axvspan(d_film_nm, z_backface_nm, alpha=0.10, color="steelblue")

# Intensity curves
ax1.plot(z_film_nm, I_film / I0, color="crimson", lw=2, label="Film")
ax1.plot(d_film_nm + z_sub_nm, I_sub / I0, color="steelblue", lw=2, label="Substrate")

# Back-face vertical step
ax1.axvline(z_backface_nm, color="navy", lw=1.5, ls="--", label=f"Back face ({d_sub_um:.0f} µm)")
# Drop dot showing back-face reflection loss
ax1.plot(
    [z_backface_nm, z_backface_nm],
    [I_sub[-1] / I0, I_sub_back / I0],
    color="navy", lw=2,
)

# Film/substrate interface
ax1.axvline(d_film_nm, color="black", lw=1.5, ls="--", label=f"Interface ({d_film_nm:.0f} nm)")

# Annotate delta inside film
if np.isfinite(delta_film_nm) and delta_film_nm < d_film_nm:
    yval_delta = (1 - R_display) * np.exp(-1)
    ax1.axvline(delta_film_nm, color="orange", lw=1, ls=":", alpha=0.9)
    ax1.annotate(
        f"δ = {delta_film_nm:.0f} nm",
        xy=(delta_film_nm, yval_delta / I0),
        xytext=(delta_film_nm + d_film_nm * 0.07, yval_delta / I0 * 1.15),
        fontsize=8, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8),
    )

# Annotate d99
if np.isfinite(d99c_nm) and d99c_nm < z_backface_nm:
    ax1.axvline(d99c_nm, color="purple", lw=1, ls=":", alpha=0.9)
    ax1.annotate(
        f"d₉₉ = {d99c_nm:.0f} nm",
        xy=(d99c_nm, 0.01),
        xytext=(d99c_nm + d_film_nm * 0.07, 0.04),
        fontsize=8, color="purple",
        arrowprops=dict(arrowstyle="->", color="purple", lw=0.8),
    )

ax1.set_xlabel("Depth z [nm]")
ax1.set_ylabel("I(z) / I₀")
ax1.set_title("Laser intensity vs depth")
if log_scale:
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=1e-4)
else:
    ax1.set_ylim(bottom=0)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

if show_gen:
    ax2 = axes[1]
    # Convert from I₀/cm to I₀/nm for nicer axis numbers
    gen_film_per_nm = gen_film * 1e-7
    gen_sub_per_nm = gen_sub * 1e-7

    ax2.axvspan(d_film_nm, z_backface_nm, alpha=0.10, color="steelblue")
    ax2.fill_between(z_film_nm, gen_film_per_nm, alpha=0.25, color="crimson")
    ax2.fill_between(d_film_nm + z_sub_nm, gen_sub_per_nm, alpha=0.25, color="steelblue")
    ax2.plot(z_film_nm, gen_film_per_nm, color="crimson", lw=2, label="Film")
    ax2.plot(d_film_nm + z_sub_nm, gen_sub_per_nm, color="steelblue", lw=2, label="Substrate")
    ax2.axvline(d_film_nm, color="black", lw=1.5, ls="--")
    ax2.axvline(z_backface_nm, color="navy", lw=1.5, ls="--")
    ax2.set_xlabel("Depth z [nm]")
    ax2.set_ylabel("Absorbed intensity [I₀/nm]")
    ax2.set_title("Where is the laser absorbed?\n(PL excitation depth profile)")
    if log_scale:
        ax2.set_yscale("log")
        peak = max(gen_film_per_nm.max() if len(gen_film_per_nm) else 1,
                   gen_sub_per_nm.max() if len(gen_sub_per_nm) else 1e-20)
        ax2.set_ylim(bottom=peak * 1e-5)
    else:
        ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()

with plot_col:
    st.pyplot(fig)
plt.close(fig)

if show_gen:
    st.info(
        "**Reading the absorption profile:** Each point shows how much laser energy is deposited "
        "per nanometre at that depth. The area under the **film curve** (pink) vs the "
        "**substrate curve** (blue) is proportional to the fraction of PL signal originating "
        "from each layer — useful for judging substrate PL contamination."
    )

if use_tmm:
    st.caption(
        "Oscillations inside the film region are **thin-film interference** (standing wave). "
        "They are most prominent when the film thickness ≈ λ/(4n) and become negligible "
        "for thick strongly-absorbing films."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.caption(
    "**v1 limitations:** Normal incidence only. Transfer Matrix Method covers coherent "
    "single-film interference but excludes multiple coherent layers (full TMM stack), "
    "incoherent multiple reflections at the back face, and non-normal incidence. "
    "See README.md for details."
)
