# Laser Penetration Depth Calculator

A Streamlit app for calculating optical penetration depth of a laser in a
thin-film / substrate stack, with visualization of intensity and carrier
generation profiles. Designed for analysis of photoluminescence (PL)
experiments with UV lasers (default: 355 nm Nd:YAG 3rd harmonic).

## Quick start

```bash
conda activate depth          # or: pip install -r requirements.txt
streamlit run app.py
```

## Features

- Absorption coefficient α from extinction coefficient k (or enter α directly)
- Normal-incidence Fresnel reflectance at air/film and film/substrate interfaces
- 1/e penetration depth δ and d₉₉ (two variants)
- Intensity profile I(z)/I₀ across film + substrate, with log-scale toggle
- Absorbed power density −dI/dz (proportional to photocarrier generation rate)
- Warning when film thickness < δ (substrate PL contribution likely)
- Material presets: TiO₂ anatase/rutile, ZnO, ZnO:Al, fused silica, glass, sapphire, Si

## Running tests

```bash
pytest test_optics.py -v
```

## Physics summary

The app implements single-pass Beer–Lambert decay with Fresnel boundary
conditions at each interface. All seven formulas are rendered in the UI via
`st.latex`.

## v1 Limitations (out of scope)

| Limitation | When it matters |
|---|---|
| **Transfer-matrix multilayer interference** | When d_film ~ λ and k is small; creates oscillating I(z) |
| **Multiple internal reflections** | High-reflectance interfaces (e.g., Si substrate) |
| **Non-normal incidence** | Oblique-angle PL collection geometries |
| **Pulsed-laser heating / ablation** | High-fluence ultrafast experiments |

Future work: implement the transfer-matrix method (TMM) for a fully rigorous
multilayer calculation, including coherent interference.
