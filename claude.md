# CLAUDE.md — Laser Penetration Depth Calculator

## Project purpose

A Streamlit app to calculate the optical penetration depth of a 355 nm laser
(Nd:YAG 3rd harmonic, default — must stay user-editable) in semiconductor
thin films (TiO2, ZnO, ZnO:Al) deposited on substrates. The tool supports the
analysis of photoluminescence (PL) emission signals: it tells the user how
deep the excitation light penetrates, what fraction is absorbed in the film
vs transmitted into the substrate, and visualizes the intensity-depth profile
across the film/substrate stack.

Python env: /opt/anaconda3/envs/depth

## Physics — formulas to implement (display each in the UI with st.latex)

1. Complex refractive index:
   n_tilde = n + i*k

2. Absorption coefficient (key formula):
   alpha = 4 * pi * k / lambda          [cm^-1, with lambda in cm]

3. Beer–Lambert intensity decay (with surface reflection):
   I(z) = I0 * (1 - R) * exp(-alpha * z)

4. Penetration depth (1/e of the transmitted intensity):
   delta = 1 / alpha = lambda / (4 * pi * k)

5. Depth at which 99% of the INCIDENT intensity is gone
   (reflection + absorption combined — see derivation note below):
   d99 = ln(100 * (1 - R)) / alpha
   Also display the simple variant d99_simple = ln(100)/alpha and label the
   difference clearly.

6. Normal-incidence Fresnel reflectance at the air/film interface:
   R = ((n - 1)^2 + k^2) / ((n + 1)^2 + k^2)

7. Film/substrate interface: when light reaches z = d_film, compute the
   reflectance at the film/substrate interface with the general two-media
   Fresnel formula:
   R_fs = ((n_f - n_s)^2 + (k_f - k_s)^2) / ((n_f + n_s)^2 + (k_f + k_s)^2)
   Transmitted fraction continues into the substrate and decays with
   alpha_substrate.

### Derivation note (keep as an expander in the UI)
The (1-R) prefactor cancels when solving for the 1/e depth delta, so delta
is independent of R. It does NOT cancel for d99 when defined relative to
the incident intensity: (1-R)*exp(-alpha*z) = 0.01 gives
z = ln(100*(1-R))/alpha, slightly shallower than ln(100)/alpha.

## Input parameters (user-provided, sidebar)

Laser:
- lambda  : wavelength [nm], default 355, editable
- I0      : incident intensity, default 1.0 (normalized) — absolute units
            optional, app works in relative units

Film (one set; consider a selectbox with presets for TiO2 anatase,
TiO2 rutile, ZnO, ZnO:Al but ALWAYS keep n, k editable):
- n_film  : real refractive index at lambda
- k_film  : extinction coefficient at lambda
- d_film  : film thickness [nm]
- Alternative input mode: let the user enter alpha_film [cm^-1] directly
  instead of k (toggle), since alpha is often what is measured.

Substrate:
- n_sub   : real refractive index at lambda
- k_sub   : extinction coefficient at lambda (0 for transparent substrates
            like fused silica/glass at 355 nm)
- material presets: fused silica, soda-lime glass, sapphire, Si — editable

## Derived quantities (computed, shown as st.metric cards)

- R (air/film), alpha_film, delta_film, d99 (both variants)
- Fraction of incident light absorbed in the film:
  A_film = (1-R) * (1 - exp(-alpha_film * d_film)) * (1 - small correction
  for interface reflection — first pass: ignore multiple reflections)
- Fraction transmitted into the substrate
- R_fs at film/substrate interface
- alpha_sub, delta_sub if k_sub > 0
- Warning badge when d_film < delta_film ("laser reaches the substrate —
  substrate PL contribution likely")

## Plot requirements (matplotlib or plotly)

- I(z)/I0 vs depth z, from z=0 through the film and into the substrate
  (continue the decay in the substrate with alpha_sub)
- Vertical dashed line + shaded region marking the film/substrate boundary
- Markers/annotations for delta (1/e) and d99
- Visible intensity discontinuity at the interface if R_fs > 0
- Log-scale toggle for the y axis
- Second optional plot: absorbed power density dI/dz vs z (this is what
  generates PL — proportional to local carrier generation rate)

## Engineering conventions

- Single file app.py is fine; if it grows, split physics into optics.py
  with pure functions (testable, no streamlit imports)
- All physics functions take SI-consistent units internally (cm), convert
  at the UI boundary; document units in every docstring
- Round displayed numbers sensibly (e.g. delta to 1 nm, R to 0.1%)
- Use st.latex for every formula listed above, next to where its result
  is displayed
- requirements.txt: streamlit, numpy, matplotlib (or plotly)
- No external API calls; fully offline tool
- Add a short "How to measure k for your own films" expander mentioning
  spectroscopic ellipsometry and the UV-Vis relation
  alpha = (1/d) * ln((1-R)^2 / T)

## Out of scope (v1)

- Full transfer-matrix multilayer interference (mention as future work in
  README; relevant when d_film ~ lambda and films are weakly absorbing)
- Multiple internal reflections / etalon effects
- Angle-dependent (non-normal) incidence
- Pulsed-laser heating or ablation physics

## Validation values (sanity checks — implement as tests if tests exist)

- ZnO: k = 0.55 at 355 nm -> alpha ≈ 1.95e5 cm^-1, delta ≈ 51 nm
- Literature reference: alpha(ZnO, 355 nm) ≈ 2e5 cm^-1
- R for n=2.0, k=0.55 -> ≈ 13%
- delta formula check: delta [nm] = lambda [nm] / (4 * pi * k)