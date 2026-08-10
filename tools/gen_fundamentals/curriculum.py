#!/usr/bin/env python3
"""
curriculum.py — Seismology fundamentals curriculum for nanoGPT-Seis-Posttrain.

This is DATA, not code: an explicit syllabus of what a domain assistant should
know before it ever sees a research paper. It exists because an audit of the
paper-derived SFT corpus found the fundamentals almost entirely absent:

    topic                                   examples   % of 4,434
    liquefaction                                 218        4.9%
    P-wave AND S-wave together                    71        1.6%
      ...with "compressional"/"longitudinal"      15        0.3%
      ...with "transverse"/particle motion         7        0.2%
    deep-focus earthquakes                        10        0.2%
      ...with the actual mechanism                 3        0.1%

Model behaviour tracked that distribution exactly: coherent on liquefaction,
incoherent on P/S waves and deep-focus earthquakes. Q&A auto-generated from
random paper excerpts teaches *specific studies*, never the underlying subject.

Each TOPIC becomes one "chunk" in the output file, so the repo's chunk-level
train/val splitter keeps a topic's questions together and cannot leak.

Edit freely: add topics, change `n_pairs`, drop sections you do not need.
"""

from __future__ import annotations
from typing import Dict, List

# Difficulty mix requested per topic. The generator asks for this many Q&A
# pairs and this blend of types.
#   conceptual    — explain a mechanism or distinction, no arithmetic
#   numerical     — worked calculation with <think> reasoning and a final value
#   comparison    — contrast two things that are commonly confused
DEFAULT_MIX = {"conceptual": 3, "numerical": 2, "comparison": 1}
CONCEPT_ONLY = {"conceptual": 4, "numerical": 0, "comparison": 2}
QUANT_HEAVY = {"conceptual": 2, "numerical": 4, "comparison": 0}


def _t(topic: str, section: str, must_cover: List[str], mix: Dict[str, int] = None,
       notes: str = "") -> Dict:
    return {
        "topic": topic,
        "section": section,
        "must_cover": must_cover,
        "mix": mix or DEFAULT_MIX,
        "notes": notes,
    }


CURRICULUM: List[Dict] = [

    # ---------------------------------------------------------------- waves
    _t("Seismic body waves: P versus S", "wave_physics",
       ["P-waves are compressional/longitudinal: particle motion parallel to propagation",
        "S-waves are shear/transverse: particle motion perpendicular to propagation",
        "Vp > Vs, typically Vp/Vs ~ 1.7-1.8 (sqrt(3) for a Poisson solid)",
        "S-waves cannot propagate through fluids (zero shear modulus) — hence the liquid outer core shadow",
        "Vp = sqrt((K + 4mu/3)/rho), Vs = sqrt(mu/rho)"],
       CONCEPT_ONLY,
       notes="This distinction was measurably missing from the corpus. Be explicit and correct."),

    _t("Surface waves: Love and Rayleigh", "wave_physics",
       ["Rayleigh waves: retrograde elliptical motion in the vertical plane, exist at a free surface",
        "Love waves: horizontal shear motion perpendicular to propagation, need a low-velocity layer",
        "Surface waves are dispersive; body waves largely are not",
        "Amplitude decays with depth; dominate long-period ground motion at regional distances",
        "Arrival order: P, then S, then surface waves"],
       CONCEPT_ONLY),

    _t("Seismic wave attenuation and Q", "wave_physics",
       ["Geometrical spreading versus intrinsic attenuation versus scattering",
        "Quality factor Q; amplitude ~ exp(-pi f t / Q)",
        "Higher frequencies attenuate faster, so distant records look long-period",
        "Typical crustal Q values and frequency dependence"],
       DEFAULT_MIX),

    _t("Earth's internal structure from seismic waves", "wave_physics",
       ["Crust, mantle, outer core, inner core and the Moho",
        "P-wave shadow zone 103-142 degrees; S-wave shadow beyond 103 degrees",
        "S-wave shadow proves the outer core is liquid",
        "PREM-style velocity increase with depth, low-velocity zone in the upper mantle"],
       CONCEPT_ONLY),

    # ------------------------------------------------------------ magnitude
    _t("Seismic moment and moment magnitude", "magnitude",
       ["M0 = mu * A * D with units N*m (mu shear modulus, A rupture area, D average slip)",
        "Mw = (2/3) log10(M0) - 6.07 for M0 in N*m",
        "Mw = (2/3) log10(M0) - 10.7 for M0 in dyne*cm — state which unit convention is in use",
        "Typical crustal mu ~ 30-33 GPa",
        "M0 is a physical quantity; Mw is a dimensionless scale derived from it"],
       QUANT_HEAVY,
       notes="The model conflates M0 and Mw and mixes the two constants. Make the unit "
             "convention explicit in every worked example."),

    _t("Magnitude scales and saturation", "magnitude",
       ["ML (Richter), mb, Ms, Mw — what each measures and its period band",
        "mb saturates near 6.5, Ms near 8.2; Mw does not saturate",
        "One magnitude unit is ~31.6x in energy, ~10x in amplitude",
        "Why Mw is preferred for large events"],
       DEFAULT_MIX),

    _t("Radiated energy and the energy-magnitude relation", "magnitude",
       ["log10(Es) = 1.5 Ms + 4.8 (Es in joules)",
        "Apparent stress and the ratio Es/M0",
        "Energy released rises ~31.6x per magnitude unit"],
       QUANT_HEAVY),

    # ---------------------------------------------------------------- source
    _t("Focal mechanisms and beachball diagrams", "source_physics",
       ["Strike, dip, rake define fault orientation and slip direction",
        "Rake ~0 left-lateral, ~180 right-lateral, ~90 reverse/thrust, ~-90 normal",
        "Two nodal planes; the fault plane ambiguity and how aftershocks resolve it",
        "Compressional (dark) versus dilatational (light) quadrants",
        "P and T axes are not the principal stress axes"],
       CONCEPT_ONLY),

    _t("The seismic moment tensor", "source_physics",
       ["Symmetric 3x3 tensor; six independent components",
        "Decomposition into isotropic, double-couple and CLVD parts",
        "A pure shear fault is a double couple with zero trace",
        "Non-double-couple components in volcanic and induced events",
        "Distinguish the moment tensor from the scalar moment M0"],
       CONCEPT_ONLY,
       notes="Model output confuses the tensor with the scalar M0. Keep them separate."),

    _t("Stress drop and rupture dimension", "source_physics",
       ["Circular crack: delta-sigma = (7/16) * M0 / r^3",
        "Typical crustal stress drops 1-10 MPa, roughly scale-invariant",
        "Distinguish static stress drop from dynamic stress drop",
        "Relation to rupture area for rectangular versus circular fault models"],
       QUANT_HEAVY,
       notes="Model emits nonsense like 'delta-sigma = (7/16) * W / W'. Every formula "
             "must have distinct, defined symbols."),

    _t("Rupture propagation and directivity", "source_physics",
       ["Rupture velocity typically 0.7-0.9 times Vs; supershear as the exception",
        "Forward directivity: shorter, larger-amplitude pulses",
        "Unilateral versus bilateral rupture",
        "Slip distribution heterogeneity: asperities and barriers"],
       DEFAULT_MIX),

    _t("Earthquake depth distribution and deep-focus earthquakes", "source_physics",
       ["Shallow < 70 km, intermediate 70-300 km, deep 300-700 km",
        "Deep-focus events occur ONLY in subducting slabs (Wadati-Benioff zones)",
        "Below ~50 km, normal brittle frictional failure is suppressed by pressure",
        "Proposed mechanisms: dehydration embrittlement (intermediate depth) and "
        "transformational faulting in metastable olivine (deep)",
        "Seismicity stops near 700 km where the slab equilibrates"],
       CONCEPT_ONLY,
       notes="CRITICAL GAP: the model claims deep-focus earthquakes occur at 10-30 km. "
             "State the 300-700 km range and the mechanism unambiguously."),

    # ------------------------------------------------------------- tectonics
    _t("Plate boundaries and their earthquake styles", "tectonics",
       ["Divergent: shallow normal faulting, mid-ocean ridges",
        "Convergent: thrust megathrusts, the largest earthquakes",
        "Transform: strike-slip, e.g. San Andreas, North Anatolian",
        "Why the greatest events (Mw > 9) occur only on subduction megathrusts"],
       CONCEPT_ONLY),

    _t("Subduction zone structure and seismogenic behaviour", "tectonics",
       ["Trench, accretionary wedge, forearc, volcanic arc, backarc",
        "Updip and downdip limits of the seismogenic zone",
        "Interplate megathrust versus intraslab versus outer-rise events",
        "Slow slip, tremor and the transition to stable sliding"],
       DEFAULT_MIX),

    _t("Elastic rebound and the earthquake cycle", "tectonics",
       ["Interseismic strain accumulation, coseismic release, postseismic relaxation",
        "Reid's elastic rebound theory",
        "Recurrence intervals and the (limited) seismic gap hypothesis",
        "Afterslip and viscoelastic relaxation"],
       CONCEPT_ONLY),

    _t("Fault friction and rate-and-state behaviour", "tectonics",
       ["Static versus dynamic friction; Byerlee's law",
        "Rate-and-state: a-b < 0 velocity weakening (seismic), a-b > 0 strengthening (aseismic)",
        "Critical slip distance Dc and nucleation size",
        "Why some faults creep and others lock"],
       DEFAULT_MIX),

    # ------------------------------------------------------------ statistics
    _t("Gutenberg-Richter and b-value", "statistics",
       ["log10 N = a - b M",
        "b is typically near 1.0; interpreting deviations",
        "a-value reflects productivity, b-value the size distribution",
        "Magnitude of completeness Mc and why it matters"],
       QUANT_HEAVY),

    _t("Aftershocks: Omori law and Bath's law", "statistics",
       ["Modified Omori: n(t) = K / (t + c)^p, p typically ~1",
        "Bath's law: largest aftershock ~1.2 magnitude units below the mainshock",
        "Aftershock zone scales with rupture length",
        "Foreshocks, swarms, and ETAS as a descriptive model"],
       QUANT_HEAVY),

    _t("Earthquake scaling relations", "statistics",
       ["Rupture length, width and area versus magnitude (Wells & Coppersmith style)",
        "Self-similarity and constant stress drop",
        "Average slip scaling with rupture dimension",
        "Saturation of rupture width in the seismogenic layer"],
       QUANT_HEAVY),

    # --------------------------------------------------------------- hazard
    _t("Ground motion intensity measures", "hazard",
       ["PGA, PGV, PGD and spectral acceleration Sa(T)",
        "Response spectra and structural period",
        "Intensity (MMI) is observational; magnitude is instrumental",
        "Duration measures and why they matter for damage"],
       DEFAULT_MIX),

    _t("Ground motion prediction equations and site effects", "hazard",
       ["GMPE dependence on magnitude, distance, style of faulting, Vs30",
        "Site amplification in soft sediments; basin effects",
        "Vs30 site classes",
        "Epistemic versus aleatory uncertainty (sigma)"],
       DEFAULT_MIX),

    _t("Probabilistic seismic hazard analysis", "hazard",
       ["The four PSHA steps: sources, recurrence, ground motion, integration",
        "Return period versus annual exceedance probability",
        "2% in 50 years = ~2475-year return period",
        "Deaggregation; PSHA versus deterministic analysis"],
       QUANT_HEAVY),

    _t("Soil liquefaction", "hazard",
       ["Saturated loose granular soils; cyclic loading raises pore pressure",
        "Effective stress falls to zero, so the soil loses shear strength",
        "Cyclic stress ratio versus cyclic resistance ratio, factor of safety",
        "Consequences: lateral spreading, settlement, bearing failure, sand boils",
        "Clay-rich soils are generally not susceptible"],
       DEFAULT_MIX),

    _t("Tsunami generation and propagation", "hazard",
       ["Seafloor displacement from megathrust ruptures",
        "Shallow-water wave speed c = sqrt(g*h)",
        "Shoaling amplification and runup",
        "Why tsunami earthquakes radiate weak short-period energy"],
       QUANT_HEAVY),

    _t("Earthquake early warning", "hazard",
       ["P-wave arrives before the damaging S and surface waves",
        "Blind zone near the epicentre; warning time versus distance",
        "Point-source versus finite-fault approaches",
        "Sources of false alarms and missed alerts"],
       DEFAULT_MIX),

    # ---------------------------------------------------------- observation
    _t("Seismometers and instrument response", "observation",
       ["Inertial pendulum principle; broadband versus short-period versus strong-motion",
        "Instrument response and why deconvolution is needed",
        "Velocity versus acceleration recording; clipping",
        "Sampling rate and the Nyquist limit"],
       DEFAULT_MIX),

    _t("Earthquake location and travel-time analysis", "observation",
       ["Four unknowns: origin time plus three hypocentre coordinates",
        "S-P time gives distance; triangulation across stations",
        "Geiger's method; the depth/origin-time trade-off",
        "Relative relocation (double-difference) and why it sharpens clusters",
        "Azimuthal gap and location uncertainty"],
       DEFAULT_MIX),

    _t("Seismic phases and travel-time curves", "observation",
       ["Direct P and S; Pg, Pn, PmP and crustal phases",
        "Core phases PKP, PKIKP; reflections pP, sP and depth phases",
        "How depth phases constrain focal depth",
        "Reading a travel-time curve"],
       CONCEPT_ONLY),

    _t("Geodetic observation of deformation", "observation",
       ["GNSS/GPS for interseismic and coseismic displacement",
        "InSAR interferograms and fringe counting",
        "Coupling ratio and slip deficit accumulation",
        "Complementarity with seismic data"],
       DEFAULT_MIX),

    # ------------------------------------------------------------ induced
    _t("Induced and triggered seismicity", "special_topics",
       ["Wastewater injection, hydraulic fracturing, reservoir impoundment, geothermal, mining",
        "Pore pressure diffusion reduces effective normal stress (Mohr-Coulomb)",
        "Distinguish induced from triggered",
        "Traffic-light protocols"],
       DEFAULT_MIX),

    _t("Coulomb stress transfer", "special_topics",
       ["delta-CFS = delta-tau + mu' * delta-sigma_n",
        "Stress shadows and increased-stress lobes",
        "Application to aftershock and fault-interaction forecasting",
        "Typical triggering thresholds ~0.01 MPa"],
       QUANT_HEAVY),

    _t("Volcano seismology", "special_topics",
       ["Volcano-tectonic events, long-period events, tremor, hybrid events",
        "Magma movement and fluid resonance",
        "Precursory swarms and eruption forecasting",
        "How these differ from tectonic earthquakes"],
       CONCEPT_ONLY),

    _t("Seismic tomography and imaging", "special_topics",
       ["Travel-time residual inversion for velocity structure",
        "Body-wave versus surface-wave tomography; ambient noise",
        "Checkerboard resolution tests",
        "Interpreting velocity anomalies as thermal or compositional"],
       DEFAULT_MIX),

    # =====================================================================
    # EXPANSION (added after measuring novelty decay across variants).
    #
    # Re-running the first 33 topics with more --variants gives diminishing
    # returns: variant 1 contributed 55% new word-types, variant 11 only 12%.
    # Extra variants paraphrase; extra TOPICS add knowledge. The blocks below
    # roughly double distinct coverage.
    # =====================================================================

    # ------------------------------------------------------- source spectra
    _t("Earthquake source spectra and corner frequency", "source_physics",
       ["Brune omega-squared model: flat at low frequency, falling as f^-2 above the corner",
        "Low-frequency spectral plateau is proportional to seismic moment",
        "Corner frequency scales inversely with source dimension",
        "Estimating stress drop from corner frequency and moment",
        "Trade-off between attenuation (Q) and corner frequency in fitting"],
       QUANT_HEAVY),

    _t("Seismic anisotropy and shear-wave splitting", "wave_physics",
       ["Anisotropy: seismic velocity depends on propagation direction",
        "A shear wave splits into fast and slow components with a delay time",
        "Fast polarization direction and delay time as the two observables",
        "Causes: aligned crystals (olivine LPO), aligned cracks, fine layering",
        "Used to infer mantle flow and crustal stress orientation"],
       DEFAULT_MIX),

    _t("Moment tensor inversion in practice", "observation",
       ["Waveform fitting versus first-motion polarity methods",
        "Green's functions and the need for a velocity model",
        "Centroid location and centroid time versus hypocentre",
        "Why shallow events have poorly constrained CLVD/isotropic components",
        "Full versus deviatoric versus double-couple-constrained inversion"],
       DEFAULT_MIX),

    _t("Okada dislocation modelling of surface deformation", "observation",
       ["Rectangular dislocation in an elastic half-space",
        "Inputs: strike, dip, rake, slip, fault dimensions, depth",
        "Predicted surface displacement field for comparison with GNSS/InSAR",
        "Elastic half-space assumptions and their limits",
        "Forward modelling versus slip inversion"],
       DEFAULT_MIX),

    # ------------------------------------------------------ signal processing
    _t("Seismic signal processing fundamentals", "observation",
       ["Sampling, aliasing and the Nyquist frequency",
        "Band-pass filtering and the trade-off between resolution and noise",
        "Fourier versus time-frequency (spectrogram, wavelet) representation",
        "Instrument response removal by deconvolution",
        "Tapering and why it matters before an FFT"],
       DEFAULT_MIX),

    _t("Automatic event detection and phase picking", "observation",
       ["STA/LTA ratio triggering and threshold selection",
        "Template matching / matched filtering for repeating events",
        "Deep-learning phase pickers and how they changed catalogue completeness",
        "False positives versus missed detections",
        "Why lowering Mc reveals more events without changing b much"],
       DEFAULT_MIX),

    _t("Ambient noise seismology", "observation",
       ["Cross-correlating ambient noise recovers the empirical Green's function",
        "Microseism peaks near 7 s and 14 s from ocean waves",
        "Surface-wave dispersion from noise, without earthquakes",
        "Monitoring temporal velocity changes (dv/v)",
        "Requirement of long recordings and diffuse wavefield"],
       CONCEPT_ONLY),

    # ------------------------------------------------------------ rock physics
    _t("Poroelasticity and fluid effects on faulting", "special_topics",
       ["Effective stress: sigma_eff = sigma - alpha * p",
        "Raising pore pressure moves the Mohr circle toward failure",
        "Undrained versus drained response; Skempton coefficient",
        "Diffusion of pore pressure and the timing of triggered events",
        "Relevance to induced seismicity and to aftershocks"],
       DEFAULT_MIX),

    _t("Rock friction experiments and laboratory earthquakes", "special_topics",
       ["Direct shear and rotary shear apparatus",
        "Slide-hold-slide tests measuring healing",
        "Velocity-step tests measuring a and b",
        "Extrapolation problem: laboratory scale versus natural faults",
        "Dynamic weakening: thermal pressurization and flash heating"],
       CONCEPT_ONLY),

    _t("Seismic velocity structure of the crust", "wave_physics",
       ["Typical Vp in sediments, upper crust, lower crust, uppermost mantle",
        "Moho as a velocity discontinuity near 7.8-8.2 km/s",
        "Vp/Vs as an indicator of fluid content or lithology",
        "Refraction versus reflection profiling",
        "Receiver functions for crustal thickness"],
       DEFAULT_MIX),

    # ------------------------------------------------------------ engineering
    _t("Response spectrum and structural dynamics", "hazard",
       ["Single-degree-of-freedom oscillator: T = 2*pi*sqrt(m/k)",
        "Building the response spectrum from a ground-motion record",
        "Damping ratio and its effect on spectral ordinates",
        "Sd = Sa * (T / 2*pi)^2 for a lightly damped system",
        "Resonance when structural period matches the dominant ground period"],
       QUANT_HEAVY),

    _t("Soil-structure interaction and site response analysis", "hazard",
       ["1-D equivalent-linear versus nonlinear site response",
        "Impedance contrast and amplification",
        "Kinematic versus inertial interaction",
        "Soft soils amplify long periods and can de-amplify short periods",
        "Quarter-wavelength approximation for site amplification"],
       DEFAULT_MIX),

    _t("Seismic design philosophy and performance levels", "hazard",
       ["Life safety versus immediate occupancy versus collapse prevention",
        "Design basis earthquake versus maximum considered earthquake",
        "Behaviour/response modification factor q or R and ductility",
        "Capacity design: strong column, weak beam",
        "Why elastic design for the largest event is uneconomic"],
       CONCEPT_ONLY),

    _t("Seismic vulnerability and fragility curves", "hazard",
       ["Fragility curve: probability of exceeding a damage state versus intensity measure",
        "Lognormal fragility parameters (median and dispersion)",
        "Empirical, analytical and expert-opinion derivation",
        "Combining fragility with hazard to get risk",
        "Expected annual loss as an integral over the hazard curve"],
       QUANT_HEAVY),

    # ------------------------------------------------------------- monitoring
    _t("Seismic networks and station siting", "observation",
       ["Network geometry, azimuthal gap and detection threshold",
        "Borehole versus surface installation and noise reduction",
        "Timing accuracy via GPS and why it matters for location",
        "Local, regional and global network roles",
        "Telemetry, latency and real-time processing"],
       DEFAULT_MIX),

    _t("Nuclear explosion monitoring and source discrimination", "special_topics",
       ["Explosions are near-isotropic; earthquakes are predominantly double-couple",
        "mb versus Ms discrimination",
        "P/S amplitude ratios at regional distances",
        "Depth of burial and the depth phase constraint",
        "The role of the CTBTO monitoring system"],
       CONCEPT_ONLY),

    _t("Earthquake catalogues and magnitude of completeness", "statistics",
       ["Catalogue construction, homogenization and magnitude conversion",
        "Estimating Mc by maximum curvature or goodness-of-fit",
        "Declustering: removing aftershocks for Poissonian hazard input",
        "Catalogue artefacts: network changes, blast contamination",
        "Why Mc varies in space and time"],
       DEFAULT_MIX),

    _t("Operational earthquake forecasting", "statistics",
       ["ETAS: background rate plus triggered aftershock cascades",
        "Short-term probability gains after a large event",
        "Why deterministic prediction has repeatedly failed",
        "Communicating low-probability, high-consequence forecasts",
        "Testing forecasts prospectively (CSEP-style)"],
       DEFAULT_MIX),

    # ---------------------------------------------------------- earth history
    _t("Paleoseismology and long earthquake records", "tectonics",
       ["Trenching to expose offset stratigraphy",
        "Radiocarbon dating of faulted horizons",
        "Turbidites and coastal subsidence as offshore records",
        "Recurrence intervals longer than the instrumental record",
        "Cascadia 1700 as a case where paleo-records fixed the date"],
       CONCEPT_ONLY),

    _t("Intraplate earthquakes and stable continental regions", "tectonics",
       ["Earthquakes far from plate boundaries: New Madrid, Charlevoix",
        "Slow strain rates and very long recurrence intervals",
        "Higher stress drops and lower attenuation than active regions",
        "Larger felt areas for the same magnitude in stable cratons",
        "Hazard estimation with sparse historical data"],
       CONCEPT_ONLY),

    _t("Volcano-tectonic interaction and dyke intrusion", "special_topics",
       ["Dyke propagation produces earthquake swarms that migrate",
        "Inflation and deflation cycles measured geodetically",
        "Distinguishing magmatic from tectonic swarms",
        "Caldera collapse earthquakes and their unusual mechanisms",
        "Triggering of eruptions by large regional earthquakes"],
       CONCEPT_ONLY),

    # --------------------------------------------------------------- coupling
    _t("Tsunami warning systems and detection", "hazard",
       ["DART buoys measuring open-ocean wave height",
        "Tide gauges and their limitations for warning",
        "Rapid magnitude estimation and the Mw saturation problem for warning",
        "Travel-time forecasting across ocean basins",
        "Why local tsunamis leave almost no warning time"],
       DEFAULT_MIX),

    _t("Earthquake-triggered landslides", "hazard",
       ["Newmark sliding-block analysis and critical acceleration",
        "Controls: slope angle, lithology, groundwater, shaking intensity",
        "Coseismic versus post-seismic landslide timing",
        "Landslide dams and their secondary hazard",
        "Topographic amplification on ridges"],
       DEFAULT_MIX),

    _t("Cryoseismology: glacial and icequake sources", "special_topics",
       ["Icequakes from crevassing, basal slip and calving",
        "Glacial earthquakes with long-period signals",
        "Distinguishing them from tectonic events by spectral content",
        "Monitoring ice-sheet dynamics seismically",
        "Glacial isostatic adjustment and induced seismicity"],
       CONCEPT_ONLY),

    _t("Slow earthquakes: tremor, LFEs and slow slip", "source_physics",
       ["The spectrum from ordinary earthquakes to slow slip events",
        "Non-volcanic tremor and low-frequency earthquakes at the plate interface",
        "Episodic tremor and slip (ETS) recurrence in Cascadia and Nankai",
        "Moment-duration scaling differs from ordinary earthquakes",
        "Possible role in loading the locked seismogenic zone"],
       DEFAULT_MIX),

    _t("Rupture directivity and near-fault ground motion", "hazard",
       ["Forward directivity produces large-amplitude, short-duration pulses",
        "Fling step from permanent tectonic displacement",
        "Fault-normal versus fault-parallel polarization",
        "Consequences for long-period structures near the fault",
        "Why directivity is hard to include in standard GMPEs"],
       DEFAULT_MIX),

    _t("Attenuation relations and regional differences", "hazard",
       ["Geometric spreading versus anelastic attenuation with distance",
        "Why eastern North America attenuates less than the west",
        "Kappa and the high-frequency spectral decay",
        "Regional adjustment (host-to-target) of GMPEs",
        "Consequences for hazard in stable continental interiors"],
       DEFAULT_MIX),

    _t("Earthquake energy budget and fracture energy", "source_physics",
       ["Partition into radiated energy, fracture energy and frictional heat",
        "Radiation efficiency and its typical values",
        "Why the absence of a heat-flow anomaly on the San Andreas is a puzzle",
        "Breakdown work and slip-weakening distance",
        "Relation between fracture energy and rupture speed"],
       DEFAULT_MIX),

    _t("Seismic moment rate and plate-boundary budgets", "tectonics",
       ["Comparing geodetic moment accumulation with seismic moment release",
        "Seismic coupling coefficient and aseismic slip",
        "Kostrov summation over a catalogue",
        "Implications when the deficit is large",
        "Sensitivity to catalogue length versus recurrence interval"],
       QUANT_HEAVY),

    _t("Building codes and seismic zonation maps", "hazard",
       ["Zonation from PSHA outputs, typically 475 or 2475 year return periods",
        "Mapped spectral accelerations at short and 1-second periods",
        "Site class adjustment factors",
        "Why codes are periodically revised as hazard models improve",
        "Difference between a hazard map and a risk map"],
       CONCEPT_ONLY),
]


def total_pairs() -> int:
    return sum(sum(t["mix"].values()) for t in CURRICULUM)


def summary() -> str:
    from collections import Counter
    sections = Counter(t["section"] for t in CURRICULUM)
    kinds = Counter()
    for t in CURRICULUM:
        for k, v in t["mix"].items():
            kinds[k] += v
    lines = [f"{len(CURRICULUM)} topics -> {total_pairs()} Q&A pairs requested", "",
             "by section:"]
    lines += [f"  {s:<16} {n:>3} topics" for s, n in sorted(sections.items())]
    lines += ["", "by question type:"]
    lines += [f"  {k:<16} {n:>3} pairs" for k, n in sorted(kinds.items())]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
