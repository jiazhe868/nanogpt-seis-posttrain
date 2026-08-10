#!/usr/bin/env python3
"""
earthquakes.py — Catalogue of historically and scientifically important
earthquakes, used as grounding for SFT data generation.

This is DATA, not code — the companion to curriculum.py. Where the curriculum
teaches settled physics, this teaches the empirical record: what happened, why
it happened, what it destroyed, and what the field learned from it.

WHY SEED FACTS MATTER HERE MORE THAN IN curriculum.py
    Fundamentals are safe to paraphrase: any correct statement of Snell's law
    is as good as another. Specific events are not. An LLM asked to write about
    the 1985 Michoacan earthquake will happily invent a magnitude, shift the
    date, or quote a death toll off by an order of magnitude, and the result
    reads exactly as fluent as the truth.

    So every entry below carries VERIFIED seed facts. They are injected into
    the prompt as the only permitted source of specifics, and the generator
    (generate_earthquakes.py) rejects any output whose stated magnitude or year
    contradicts them. Facts that are genuinely disputed in the literature
    (several death tolls) are written as ranges and flagged, so the model is
    taught the uncertainty rather than a false precision.

FIELDS
    name        canonical event name
    date        ISO date (UTC)
    mw          moment magnitude (the value to enforce; ~ prefix = approximate
                for pre-instrumental or contested estimates)
    depth_km    hypocentral depth
    region      country / tectonic region
    mechanism   faulting style
    setting     tectonic context
    facts       verified specifics the answers may use
    significance  why this event matters to the science
    mix         question-type counts for generation
"""

from __future__ import annotations
from typing import Dict, List

# Question-type mixes (mirrors curriculum.py conventions)
#   factual      — what happened: parameters, sequence of events
#   mechanism    — the physics: why this rupture, this setting, this behaviour
#   impact       — damage, casualties, engineering and societal consequences
#   significance — what the event changed in the science or in practice
#   numerical    — a calculation grounded in the event's real parameters
STANDARD = {"factual": 1, "mechanism": 2, "impact": 1, "significance": 1, "numerical": 1}
PHYSICS_HEAVY = {"factual": 1, "mechanism": 3, "impact": 0, "significance": 1, "numerical": 1}
IMPACT_HEAVY = {"factual": 1, "mechanism": 1, "impact": 3, "significance": 1, "numerical": 0}
BRIEF = {"factual": 1, "mechanism": 1, "impact": 1, "significance": 1, "numerical": 0}


def _e(name, date, mw, depth_km, region, mechanism, setting, facts, significance,
       mix=None) -> Dict:
    return {
        "name": name, "date": date, "mw": mw, "depth_km": depth_km,
        "region": region, "mechanism": mechanism, "setting": setting,
        "facts": facts, "significance": significance, "mix": mix or STANDARD,
    }


EARTHQUAKES: List[Dict] = [

    # ================================================================
    # GREAT SUBDUCTION EARTHQUAKES (Mw >= 8.5)
    # ================================================================
    _e("1960 Valdivia (Great Chilean) earthquake", "1960-05-22", 9.5, 25,
       "Chile", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["The largest earthquake ever recorded instrumentally, Mw 9.5",
        "Rupture length roughly 1000 km along the Chilean margin",
        "Generated a Pacific-wide tsunami that killed people in Hawaii, Japan and the Philippines",
        "Triggered the eruption of Cordon Caulle volcano two days later",
        "Coastal subsidence of up to about 2 m was measured"],
       "Set the upper bound on observed earthquake size and demonstrated that "
       "subduction megathrusts can rupture for a thousand kilometres. Its free "
       "oscillations were the first clear observation of the Earth's normal modes.",
       PHYSICS_HEAVY),

    _e("1964 Great Alaska (Good Friday) earthquake", "1964-03-27", 9.2, 25,
       "Alaska, USA", "megathrust (thrust)", "Aleutian subduction zone",
       ["Second largest instrumentally recorded earthquake, Mw 9.2",
        "About 4.8 minutes of shaking",
        "Caused widespread submarine and subaerial landslides",
        "Turnagain Heights landslide destroyed a residential district in Anchorage",
        "Most of the roughly 130 deaths were caused by the tsunami, not shaking",
        "Produced permanent vertical land-level changes of up to about 11 m uplift"],
       "The measured pattern of uplift and subsidence matched elastic rebound on a "
       "shallow-dipping megathrust, providing direct evidence for subduction and "
       "helping establish plate tectonics. It also motivated modern landslide and "
       "tsunami hazard assessment in the United States.",
       STANDARD),

    _e("2004 Sumatra-Andaman earthquake and Indian Ocean tsunami", "2004-12-26", 9.1, 30,
       "Indonesia / Indian Ocean", "megathrust (thrust)", "Sunda subduction zone",
       ["Rupture propagated roughly 1300 km northward over about 8-10 minutes",
        "Estimates of Mw range from about 9.1 to 9.3 depending on the period band used",
        "Approximately 230,000 deaths across fourteen countries",
        "No Indian Ocean tsunami warning system existed at the time",
        "Excited the Earth's normal modes strongly enough to be observed for weeks"],
       "The deadliest tsunami in recorded history. It exposed the absence of warning "
       "infrastructure in the Indian Ocean, drove its creation, and showed that "
       "magnitude estimates from short-period data badly underestimate the largest events.",
       IMPACT_HEAVY),

    _e("2011 Tohoku-oki earthquake and tsunami", "2011-03-11", 9.0, 29,
       "Japan", "megathrust (thrust)", "Japan Trench, Pacific-Okhotsk subduction",
       ["Mw 9.0-9.1, the largest recorded earthquake in Japan",
        "Peak coseismic slip exceeded 50 m near the trench, unusually shallow slip",
        "Tsunami runup reached about 40 m at Ryori Bay, Iwate Prefecture",
        "Approximately 19,700 deaths, the overwhelming majority from the tsunami",
        "Caused the Fukushima Daiichi nuclear accident when the tsunami disabled cooling",
        "Recorded by the dense K-NET, KiK-net and GEONET networks, plus seafloor geodesy"],
       "The best-instrumented great earthquake in history. It overturned the assumption "
       "that the shallowest part of the megathrust slips aseismically, invalidated hazard "
       "maps that capped the region near Mw 8, and reshaped tsunami and nuclear siting policy.",
       STANDARD),

    _e("2010 Maule, Chile earthquake", "2010-02-27", 8.8, 35,
       "Chile", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["Ruptured roughly 500 km of the margin",
        "Filled the Darwin seismic gap, unbroken since 1835",
        "About 500 deaths, low for the magnitude, largely thanks to strong building codes",
        "Generated a Pacific tsunami and locally damaging runup",
        "Well captured by pre-existing continuous GPS networks"],
       "A controlled comparison with Haiti six weeks earlier: an event roughly 500 times "
       "larger in energy caused a small fraction of the deaths, demonstrating that "
       "construction quality, not magnitude, dominates earthquake mortality.",
       STANDARD),

    _e("2012 Indian Ocean (off Sumatra) strike-slip earthquake", "2012-04-11", 8.6, 20,
       "Indian Ocean", "strike-slip", "oceanic intraplate, diffuse India-Australia boundary",
       ["The largest strike-slip earthquake ever recorded, Mw 8.6",
        "Occurred within the oceanic plate, not on the subduction interface",
        "Involved rupture on multiple conjugate faults",
        "Produced only a small tsunami despite its size, because slip was mostly horizontal",
        "Triggered elevated seismicity worldwide for several days"],
       "Showed that intraplate oceanic lithosphere can host earthquakes approaching Mw 9, "
       "that ruptures can cascade across conjugate fault sets, and that strike-slip events "
       "are relatively inefficient tsunami sources.",
       PHYSICS_HEAVY),

    _e("2005 Nias-Simeulue earthquake", "2005-03-28", 8.6, 30,
       "Indonesia", "megathrust (thrust)", "Sunda subduction zone",
       ["Occurred three months after the 2004 Sumatra-Andaman earthquake",
        "Ruptured the adjacent segment immediately to the southeast",
        "Coral microatolls recorded metre-scale uplift",
        "About 1300 deaths, far fewer than in 2004"],
       "A textbook case of stress transfer loading an adjacent segment, and one of the "
       "clearest demonstrations that coral microatolls can record coseismic uplift and "
       "reconstruct pre-instrumental earthquake cycles.",
       STANDARD),

    _e("2007 Pisco, Peru earthquake", "2007-08-15", 8.0, 39,
       "Peru", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["Ruptured in two distinct subevents about 60 seconds apart",
        "About 500 deaths",
        "Caused extensive liquefaction and a local tsunami",
        "Long duration of shaking relative to typical Mw 8 events"],
       "The clear two-subevent rupture made it a widely used case study for finite-fault "
       "inversion and for how asperity distribution controls ground-motion duration.",
       BRIEF),

    # ================================================================
    # DEEP-FOCUS AND INTRASLAB
    # ================================================================
    _e("2013 Sea of Okhotsk deep-focus earthquake", "2013-05-24", 8.3, 609,
       "Russia (Sea of Okhotsk)", "normal faulting within the slab", "deep subducted slab",
       ["The largest deep-focus earthquake ever recorded, Mw 8.3 at 609 km depth",
        "Ruptured with an unusually high rupture velocity, estimated near 4 km/s",
        "Felt as far away as Moscow, roughly 6000 km distant",
        "Caused no significant damage because of its great depth",
        "Radiated energy comparable to shallow events of similar moment"],
       "At 609 km, ordinary brittle friction is impossible, so this event is a key "
       "constraint on transformational faulting in metastable olivine. Its high rupture "
       "velocity and efficiency challenged models of deep rupture as intrinsically slow.",
       PHYSICS_HEAVY),

    _e("1994 Bolivia deep-focus earthquake", "1994-06-09", 8.2, 631,
       "Bolivia", "normal faulting within the slab", "deep subducted Nazca slab",
       ["Depth of about 631 km, among the deepest large earthquakes recorded",
        "Rupture area of roughly 40 x 50 km, compact for the magnitude",
        "Very low rupture velocity and high fracture energy inferred",
        "Felt in Canada, more than 8000 km away"],
       "The benchmark deep-focus earthquake. Its large fracture energy and compact rupture "
       "supported transformational faulting rather than dehydration embrittlement as the "
       "mechanism at these depths.",
       PHYSICS_HEAVY),

    _e("2017 Puebla (Mexico City) earthquake", "2017-09-19", 7.1, 57,
       "Mexico", "normal faulting within the slab", "intraslab, subducted Cocos plate",
       ["An intraslab normal-faulting event, not a megathrust earthquake",
        "Struck on the 32nd anniversary of the 1985 Michoacan earthquake",
        "About 370 deaths, most in Mexico City",
        "Shorter period ground motion than 1985, damaging mid-rise rather than tall buildings",
        "Occurred roughly 120 km from Mexico City"],
       "Demonstrated that the resonance period of the Mexico City lake-bed sediments "
       "interacts differently with intraslab and megathrust sources: the damaging building "
       "height class differed completely from 1985 despite the same city.",
       IMPACT_HEAVY),

    _e("2018 Anchorage earthquake", "2018-11-30", 7.1, 47,
       "Alaska, USA", "normal faulting within the slab", "intraslab, subducted Pacific plate",
       ["Strong shaking directly beneath a city of about 300,000 people",
        "No deaths, and comparatively limited structural damage",
        "Extensive damage to roads and non-structural building elements"],
       "A direct demonstration of the value of modern seismic codes: an event that would "
       "have been catastrophic in a poorly built city produced zero fatalities.",
       IMPACT_HEAVY),

    # ================================================================
    # CONTINENTAL STRIKE-SLIP
    # ================================================================
    _e("1906 San Francisco earthquake", "1906-04-18", "~7.9", 8,
       "California, USA", "right-lateral strike-slip", "San Andreas transform fault",
       ["Surface rupture extended about 477 km along the San Andreas Fault",
        "Maximum measured horizontal offset was roughly 6 m",
        "Most of the destruction in San Francisco came from the fire that followed",
        "About 3000 deaths",
        "Pre-instrumental: the magnitude is estimated, commonly quoted near Mw 7.9"],
       "The geodetic survey of its offsets led Harry Fielding Reid to formulate the elastic "
       "rebound theory, the foundation of the modern understanding of the earthquake cycle.",
       STANDARD),

    _e("1992 Landers earthquake", "1992-06-28", 7.3, 7,
       "California, USA", "right-lateral strike-slip", "Eastern California Shear Zone",
       ["Ruptured at least five separate fault segments in a single cascading event",
        "Maximum surface offset of about 6 m",
        "Triggered increased seismicity across the western United States, as far as Yellowstone",
        "One of the first earthquakes studied with InSAR"],
       "Established two ideas now central to the field: ruptures can jump between "
       "non-contiguous faults, and static and dynamic stress changes trigger seismicity at "
       "remote distances. It effectively launched Coulomb stress transfer modelling.",
       PHYSICS_HEAVY),

    _e("1999 Hector Mine earthquake", "1999-10-16", 7.1, 5,
       "California, USA", "right-lateral strike-slip", "Eastern California Shear Zone",
       ["Occurred in a sparsely populated area, causing minimal damage",
        "Located close to the 1992 Landers rupture",
        "Imaged with high-quality InSAR interferograms",
        "Consistent with Coulomb stress increase from the Landers event"],
       "The cleanest early InSAR case study of a continental earthquake, and strong "
       "supporting evidence for Coulomb stress triggering between neighbouring faults.",
       BRIEF),

    _e("2019 Ridgecrest earthquake sequence", "2019-07-06", 7.1, 8,
       "California, USA", "strike-slip on conjugate faults", "Eastern California Shear Zone",
       ["A foreshock of Mw 6.4 on 4 July preceded the Mw 7.1 mainshock on 6 July",
        "Rupture occurred on orthogonal conjugate faults",
        "Captured by dense modern geodetic and seismic instrumentation",
        "Surface rupture mapped in detail with satellite imagery and lidar"],
       "The best-recorded conjugate-faulting sequence to date. It challenged the assumption "
       "that mapped faults capture the hazard, since much of the rupture occurred on "
       "structures not previously identified.",
       STANDARD),

    _e("1999 Izmit (Kocaeli), Turkey earthquake", "1999-08-17", 7.6, 15,
       "Turkey", "right-lateral strike-slip", "North Anatolian Fault",
       ["Surface rupture of about 145 km",
        "Roughly 17,000 deaths, with widespread collapse of reinforced concrete buildings",
        "Struck at 03:02 local time when most people were asleep indoors",
        "Continued the westward progression of large ruptures along the North Anatolian Fault since 1939"],
       "A defining example of the westward migrating earthquake sequence on the North "
       "Anatolian Fault, and of how non-ductile reinforced concrete construction converts "
       "moderate shaking into mass casualties.",
       IMPACT_HEAVY),

    _e("1999 Chi-Chi, Taiwan earthquake", "1999-09-21", 7.6, 8,
       "Taiwan", "thrust with a strike-slip component", "Chelungpu fault, Taiwan orogen",
       ["Surface rupture of about 100 km along the Chelungpu fault",
        "Vertical offsets locally exceeded 8 m, creating waterfalls where rivers were cut",
        "About 2400 deaths",
        "Recorded by a dense strong-motion network producing hundreds of near-fault records",
        "Notably large displacements with relatively low high-frequency ground motion"],
       "Produced the richest near-fault strong-motion dataset then available and became a "
       "cornerstone for studying rupture directivity, fling step, and hanging-wall effects.",
       STANDARD),

    _e("2023 Kahramanmaras (Turkey-Syria) earthquake doublet", "2023-02-06", 7.8, 10,
       "Turkey / Syria", "left-lateral strike-slip", "East Anatolian Fault system",
       ["An Mw 7.8 mainshock was followed about nine hours later by an Mw 7.5 event on a separate fault",
        "Combined death toll exceeded 50,000 across Turkey and Syria",
        "Surface rupture totalling several hundred kilometres",
        "Widespread pancake collapse of mid-rise reinforced concrete buildings",
        "Extensive damage in a region with a long historical record of large earthquakes"],
       "One of the deadliest earthquakes of the 21st century, and a stark case of a "
       "well-mapped, well-understood fault system producing a foreseeable event whose "
       "consequences were driven overwhelmingly by building stock and code enforcement.",
       IMPACT_HEAVY),

    _e("2018 Palu, Sulawesi earthquake and tsunami", "2018-09-28", 7.5, 20,
       "Indonesia", "left-lateral strike-slip", "Palu-Koro fault",
       ["Evidence for supershear rupture, propagating faster than the shear-wave speed",
        "Generated an unexpectedly large local tsunami for a strike-slip event",
        "Catastrophic liquefaction flowslides at Petobo and Balaroa moved material long distances",
        "About 4300 deaths"],
       "Rewrote expectations twice: strike-slip earthquakes were not thought to generate "
       "large tsunamis (submarine landslides and narrow bay geometry did it here), and the "
       "flowslides showed liquefaction can mobilise gentle slopes catastrophically.",
       STANDARD),

    # ================================================================
    # BLIND THRUST AND URBAN MODERATE EVENTS
    # ================================================================
    _e("1994 Northridge earthquake", "1994-01-17", 6.7, 19,
       "California, USA", "blind thrust", "Los Angeles basin, no surface rupture",
       ["Occurred on a previously unrecognised blind thrust fault",
        "Produced no surface rupture",
        "Peak ground accelerations locally exceeded 1.0 g",
        "About 57 deaths but economic losses of roughly 20-40 billion USD",
        "Revealed brittle fracture in welded steel moment-frame connections"],
       "Among the costliest US natural disasters. It proved that unmapped blind thrusts "
       "underlie major cities, and the discovery of widespread weld fractures in steel "
       "frames forced a fundamental revision of steel connection design.",
       IMPACT_HEAVY),

    _e("1989 Loma Prieta earthquake", "1989-10-17", 6.9, 18,
       "California, USA", "oblique reverse-slip", "San Andreas system, Santa Cruz Mountains",
       ["About 63 deaths, many in the Cypress Street Viaduct collapse in Oakland",
        "Severe damage in the Marina District, built on artificial fill, tens of kilometres from the epicentre",
        "Occurred during a nationally televised World Series game",
        "Collapse of a section of the San Francisco-Oakland Bay Bridge"],
       "The Marina District damage is the standard demonstration of site amplification on "
       "soft artificial fill, and the freeway collapses drove a nationwide programme of "
       "bridge seismic retrofitting.",
       IMPACT_HEAVY),

    _e("1995 Kobe (Hyogo-ken Nanbu) earthquake", "1995-01-17", 6.9, 16,
       "Japan", "right-lateral strike-slip", "Nojima fault, Japan inland",
       ["About 6400 deaths in a modern industrialised city",
        "Massive damage to elevated expressways, port facilities and older wooden housing",
        "Extensive liquefaction of reclaimed land in Kobe Port",
        "Fires broke out and were difficult to fight because water mains had ruptured",
        "Direct economic loss estimated near 100 billion USD"],
       "Shattered confidence that a modern industrialised nation was safe from urban "
       "earthquake catastrophe. It triggered sweeping revision of Japanese design codes, "
       "large-scale retrofitting, and reorganisation of emergency response.",
       IMPACT_HEAVY),

    _e("2011 Christchurch earthquake", "2011-02-22", 6.2, 5,
       "New Zealand", "oblique reverse", "previously unmapped fault beneath the city",
       ["Technically an aftershock of the 2010 Mw 7.1 Darfield earthquake",
        "185 deaths, most in the collapse of the Canterbury Television building",
        "Vertical peak ground accelerations locally exceeded 2 g, exceptionally high",
        "Catastrophic and repeated liquefaction across large areas of the city",
        "Roughly 80 percent of the central business district was eventually demolished"],
       "Showed that a moderate, very shallow, very close earthquake can exceed a much larger "
       "distant one in destructiveness, and produced the most comprehensively documented "
       "urban liquefaction dataset in existence.",
       IMPACT_HEAVY),

    _e("1985 Michoacan (Mexico City) earthquake", "1985-09-19", 8.0, 15,
       "Mexico", "megathrust (thrust)", "Cocos-North America subduction zone",
       ["Epicentre roughly 350 km from Mexico City, yet the city suffered the worst damage",
        "Soft lake-bed clays beneath the city amplified ground motion by factors of 10-50",
        "The basin resonated near a 2 second period, matching 6-15 storey buildings",
        "At least 5000 and possibly over 10,000 deaths",
        "Buildings in the resonant height range collapsed while taller and shorter ones survived"],
       "The definitive case study of basin resonance and site effects. Damage selective by "
       "building height at 350 km distance made period-dependent site amplification "
       "impossible to ignore in any modern building code.",
       STANDARD),

    _e("2016 Kumamoto earthquake sequence", "2016-04-16", 7.0, 12,
       "Japan", "right-lateral strike-slip", "Futagawa-Hinagu fault zone",
       ["An Mw 6.2 foreshock on 14 April was followed by the larger Mw 7.0 mainshock on 16 April",
        "Many people had returned home after the foreshock and were caught by the mainshock",
        "Surface rupture crossed Kumamoto Castle and nearby infrastructure",
        "Extensive landslides in the Aso caldera region"],
       "A hard lesson in foreshock ambiguity: the first event was widely treated as the "
       "mainshock. It also demonstrated the practical difficulty of communicating that a "
       "larger event may still follow.",
       STANDARD),

    # ================================================================
    # VULNERABILITY-DOMINATED DISASTERS
    # ================================================================
    _e("2010 Haiti earthquake", "2010-01-12", 7.0, 13,
       "Haiti", "strike-slip with a reverse component", "Enriquillo-Plantain Garden fault zone",
       ["Struck about 25 km from Port-au-Prince at very shallow depth",
        "Death toll estimates vary widely, from roughly 100,000 to over 300,000",
        "Almost total collapse of unreinforced masonry and poorly built concrete structures",
        "Destroyed government buildings, crippling the national response",
        "Only a moderate magnitude event, yet among the deadliest ever recorded"],
       "The clearest demonstration that earthquake mortality is governed by construction "
       "quality and governance rather than magnitude. Compare directly with 2010 Maule, "
       "roughly 500 times more energetic with about 1 percent of the deaths.",
       IMPACT_HEAVY),

    _e("2008 Wenchuan (Sichuan) earthquake", "2008-05-12", 7.9, 19,
       "China", "thrust with a strike-slip component", "Longmenshan fault zone, eastern Tibetan Plateau margin",
       ["Surface rupture of about 240 km along the Longmenshan fault",
        "About 87,000 deaths including those missing",
        "Tens of thousands of coseismic landslides, which caused a large share of the deaths",
        "Formation of numerous landslide-dammed 'quake lakes' posing a secondary flood hazard",
        "Collapse of many school buildings became a major public controversy"],
       "The most destructive landslide-triggering earthquake on record, and the event that "
       "made coseismic landsliding a first-order component of mountain-region seismic hazard "
       "assessment rather than an afterthought.",
       IMPACT_HEAVY),

    _e("2005 Kashmir earthquake", "2005-10-08", 7.6, 26,
       "Pakistan / India", "thrust", "Himalayan collision zone",
       ["About 86,000 deaths, concentrated in mountainous terrain",
        "Widespread landslides blocked access to affected valleys",
        "Onset of Himalayan winter severely hampered relief efforts",
        "Extensive collapse of stone and unreinforced masonry housing"],
       "Demonstrated how mountainous terrain compounds an earthquake disaster: landsliding, "
       "severed access routes and an approaching winter turned a survivable event into a "
       "prolonged humanitarian emergency.",
       IMPACT_HEAVY),

    _e("2003 Bam, Iran earthquake", "2003-12-26", 6.6, 9,
       "Iran", "strike-slip", "Bam fault, Iranian plateau",
       ["About 26,000 deaths in a city of roughly 100,000",
        "Nearly complete destruction of traditional adobe and mud-brick construction",
        "Destroyed the ancient Arg-e Bam citadel, a UNESCO World Heritage site",
        "A moderate magnitude event with a catastrophic outcome"],
       "The canonical case of adobe vulnerability: heavy earthen roofs on weak walls "
       "collapse onto occupants. It drove international attention to retrofitting "
       "vernacular earthen construction.",
       IMPACT_HEAVY),

    _e("1988 Spitak, Armenia earthquake", "1988-12-07", 6.8, 5,
       "Armenia", "thrust", "Caucasus collision zone",
       ["About 25,000 deaths",
        "Precast concrete frame buildings failed catastrophically at their connections",
        "Older stone masonry buildings also collapsed widely",
        "Winter conditions complicated rescue and shelter"],
       "Exposed the fatal weakness of poorly detailed precast concrete construction, whose "
       "connections lacked ductility and continuity, prompting reassessment of that building "
       "type across the former Soviet Union.",
       IMPACT_HEAVY),

    _e("1976 Tangshan earthquake", "1976-07-28", "~7.6", 12,
       "China", "strike-slip", "intraplate, North China Plain",
       ["Struck a densely populated industrial city at 03:42 local time",
        "Official death toll about 242,000, with higher unofficial estimates",
        "Almost all unreinforced masonry buildings in the city collapsed",
        "Occurred with no recognised precursory warning",
        "One of the deadliest earthquakes of the twentieth century"],
       "A defining event for Chinese earthquake engineering and for the limits of earthquake "
       "prediction, coming shortly after the partially successful Haicheng evacuation created "
       "unrealistic expectations that prediction was a solved problem.",
       IMPACT_HEAVY),

    _e("1970 Ancash, Peru earthquake", "1970-05-31", 7.9, 45,
       "Peru", "normal faulting within the slab", "intraslab, subducted Nazca plate",
       ["Triggered a catastrophic rock and ice avalanche from Nevado Huascaran",
        "The avalanche buried the towns of Yungay and Ranrahirca",
        "Total deaths estimated near 70,000, the deadliest earthquake in Peruvian history",
        "The avalanche travelled at extremely high speed over about 16 km"],
       "The deadliest earthquake-triggered landslide ever recorded, and the standard case "
       "study for long-runout rock-ice avalanches as a cascading earthquake hazard.",
       IMPACT_HEAVY),

    _e("1923 Great Kanto earthquake", "1923-09-01", "~7.9", 23,
       "Japan", "megathrust (thrust)", "Sagami Trough, Philippine Sea plate subduction",
       ["About 105,000 deaths",
        "Struck near midday when cooking fires were lit, igniting a firestorm",
        "A fire tornado in Tokyo killed tens of thousands of people in a single location",
        "Destroyed much of Tokyo and Yokohama"],
       "The deadliest earthquake in Japanese history, and the origin of Japan's Disaster "
       "Prevention Day. It made post-earthquake fire, rather than shaking itself, a central "
       "concern of urban earthquake planning.",
       IMPACT_HEAVY),

    # ================================================================
    # HIMALAYAN AND CONTINENTAL COLLISION
    # ================================================================
    _e("2015 Gorkha, Nepal earthquake", "2015-04-25", 7.8, 15,
       "Nepal", "thrust", "Main Himalayan Thrust",
       ["About 9000 deaths",
        "Ruptured a portion of the Main Himalayan Thrust but did not break the surface",
        "Triggered a deadly avalanche at Everest Base Camp",
        "Caused smooth, long-period shaking in the Kathmandu basin that spared many low buildings",
        "Followed by a large Mw 7.3 aftershock on 12 May"],
       "The first great Himalayan earthquake captured by modern geodesy and seismology. Its "
       "unexpectedly smooth, long-period ground motion, and its failure to rupture the "
       "surface, left the shallowest segment still loaded and capable of a future event.",
       STANDARD),

    _e("2001 Bhuj (Gujarat), India earthquake", "2001-01-26", 7.6, 16,
       "India", "reverse", "intraplate, Kachchh rift basin",
       ["A large intraplate earthquake far from any plate boundary",
        "About 20,000 deaths",
        "Felt over an exceptionally large area because of low intraplate attenuation",
        "Caused extensive liquefaction in the Rann of Kachchh",
        "No surface rupture despite the large magnitude"],
       "One of the largest intraplate earthquakes ever recorded. Its enormous felt area "
       "illustrated low attenuation in stable continental crust and forced a revision of "
       "Indian seismic zoning far from plate boundaries.",
       STANDARD),

    _e("2016 Central Italy (Amatrice-Norcia) sequence", "2016-08-24", 6.2, 8,
       "Italy", "normal faulting", "Apennine extensional belt",
       ["An Mw 6.2 event in August destroyed Amatrice, killing about 300 people",
        "A larger Mw 6.5 event followed near Norcia on 30 October",
        "The later, larger event caused far fewer deaths because the area had been evacuated",
        "Severe damage to historic stone masonry towns and cultural heritage"],
       "Illustrated both the vulnerability of historic masonry centres and the value of "
       "evacuation: the larger October event killed almost nobody because people had already "
       "left damaged buildings.",
       IMPACT_HEAVY),

    _e("2009 L'Aquila earthquake", "2009-04-06", 6.3, 9,
       "Italy", "normal faulting", "Apennine extensional belt",
       ["About 309 deaths in a historic city",
        "Preceded by a months-long seismic swarm",
        "Public reassurance before the mainshock led to manslaughter convictions of scientists and officials, later largely overturned on appeal for the scientists",
        "Collapse of both historic masonry and some modern reinforced concrete buildings"],
       "The most consequential case in the history of earthquake science communication. It "
       "reshaped how scientific advisory bodies phrase probabilistic statements and clarified "
       "the distinction between failing to predict and communicating risk poorly.",
       IMPACT_HEAVY),

    # ================================================================
    # INDUCED, MONITORING AND SPECIAL CASES
    # ================================================================
    _e("2016 Pawnee, Oklahoma earthquake", "2016-09-03", 5.8, 5,
       "Oklahoma, USA", "strike-slip", "intraplate, induced by wastewater injection",
       ["The largest recorded earthquake in Oklahoma history",
        "Occurred amid a dramatic rise in Oklahoma seismicity linked to wastewater disposal",
        "Oklahoma went from a few M3+ events per year to hundreds after 2009",
        "Prompted regulatory limits on injection volumes, after which rates declined"],
       "The clearest large-magnitude example of induced seismicity from wastewater "
       "injection, and a case where regulatory intervention measurably reduced earthquake "
       "rates, demonstrating the hazard was controllable.",
       STANDARD),

    _e("Groningen induced seismicity, Netherlands", "2012-08-16", 3.6, 3,
       "Netherlands", "normal faulting", "induced by natural gas extraction",
       ["Induced by compaction of the Groningen gas field, one of the largest in the world",
        "The Mw 3.6 Huizinge event in 2012 was the largest and a turning point",
        "Very shallow depth meant even small events caused building damage",
        "Widespread damage claims on unreinforced masonry housing",
        "Led to a decision to phase out production from the field"],
       "The clearest case of induced seismicity driving a national energy policy reversal, "
       "and a demonstration that very shallow, small-magnitude events can be damaging where "
       "the building stock is unreinforced masonry.",
       IMPACT_HEAVY),

    _e("2011 Mineral, Virginia earthquake", "2011-08-23", 5.8, 6,
       "Virginia, USA", "reverse", "intraplate, Central Virginia Seismic Zone",
       ["Felt by more people than any other earthquake in US history",
        "Shaking was felt from Georgia to Canada",
        "Damaged the Washington Monument and the National Cathedral",
        "A moderate event with an enormous felt area"],
       "The standard illustration of low attenuation in eastern North America: an event that "
       "would have been barely noticed outside the epicentral region in California was felt "
       "across a third of a continent.",
       STANDARD),

    _e("2004 Parkfield earthquake", "2004-09-28", 6.0, 8,
       "California, USA", "right-lateral strike-slip", "San Andreas transform, transition to creeping section",
       ["The target of a long-running earthquake prediction experiment begun in 1985",
        "A repeat had been forecast for the late 1980s but arrived in 2004, far later than expected",
        "Rupture propagated in the opposite direction to the 1966 event",
        "Densely instrumented before it occurred, a rarity"],
       "The most important negative result in earthquake prediction. The characteristic "
       "earthquake model failed on its best test case, and the dense pre-event "
       "instrumentation showed no clear precursors.",
       STANDARD),

    _e("2010 Darfield (Canterbury), New Zealand earthquake", "2010-09-04", 7.1, 11,
       "New Zealand", "strike-slip with a reverse component", "previously unknown Greendale fault",
       ["Occurred on a fault not previously mapped, beneath farmland",
        "No deaths, largely because it struck at 04:35 when people were at home in bed and buildings performed adequately",
        "Produced clear surface rupture across the Canterbury Plains",
        "Initiated a long aftershock sequence including the deadly February 2011 Christchurch event"],
       "Showed that unmapped faults beneath low-relief agricultural terrain can produce Mw 7 "
       "earthquakes, and its aftershock sequence became a landmark case in the difficulty of "
       "forecasting damaging aftershocks.",
       STANDARD),

    # ==================================================================
    # EXPANSION — the catalogue above covers the canonical cases; these
    # broaden coverage across regions, mechanisms and decades. Depths are
    # approximate (the prompt says so), magnitudes and dates are not.
    # ==================================================================

    # ---------------------------------------------- more great subduction
    _e("1952 Kamchatka earthquake", "1952-11-04", 9.0, 30,
       "Russia (Kamchatka)", "megathrust (thrust)", "Kuril-Kamchatka subduction zone",
       ["One of only a handful of recorded Mw 9 class earthquakes",
        "Generated a Pacific-wide tsunami that damaged Hawaii",
        "Occurred before the global standardised seismograph network existed",
        "Magnitude was revised upward as long-period analysis methods improved"],
       "A reminder that the pre-1960s instrumental record systematically underestimated "
       "great earthquakes, because the available instruments could not capture the "
       "long-period energy that dominates Mw 9 events.",
       BRIEF),

    _e("1946 Aleutian Islands earthquake and tsunami", "1946-04-01", 8.6, 25,
       "Alaska, USA", "megathrust (thrust)", "Aleutian subduction zone",
       ["Generated a tsunami that destroyed the Scotch Cap lighthouse in Alaska",
        "The tsunami killed about 160 people in Hilo, Hawaii, hours later with no warning",
        "Tsunami runup was far larger than the magnitude alone would suggest",
        "Regarded as a 'tsunami earthquake' with anomalously slow rupture"],
       "Directly caused the creation of the Pacific Tsunami Warning System, and became the "
       "type example of a tsunami earthquake: slow rupture radiating weak short-period "
       "energy while displacing enormous volumes of water.",
       STANDARD),

    _e("1965 Rat Islands earthquake", "1965-02-04", 8.7, 30,
       "Alaska, USA", "megathrust (thrust)", "Aleutian subduction zone",
       ["Among the largest earthquakes of the twentieth century",
        "Occurred in a remote part of the Aleutian arc, causing little damage",
        "Generated a tsunami of about 10 m runup on Shemya Island"],
       "Helped define the along-strike segmentation of the Aleutian megathrust, showing "
       "which sections rupture together and which act as persistent barriers.",
       BRIEF),

    _e("2015 Illapel, Chile earthquake", "2015-09-16", 8.3, 22,
       "Chile", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["Generated a Pacific tsunami with local runup of several metres",
        "About 15 deaths, remarkably few for the magnitude",
        "Roughly a million people were evacuated from the coast before the tsunami arrived",
        "Struck a segment that had previously ruptured in 1943"],
       "A model case of effective tsunami warning and evacuation culture: an Mw 8.3 event "
       "with a large tsunami produced a death toll in the tens rather than the thousands.",
       STANDARD),

    _e("2014 Iquique, Chile earthquake", "2014-04-01", 8.2, 25,
       "Chile", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["Preceded by an unusually clear two-week foreshock sequence including an Mw 6.7",
        "Ruptured only part of the northern Chile seismic gap unbroken since 1877",
        "About 6 deaths",
        "Left much of the identified gap still unruptured"],
       "One of the clearest recorded foreshock sequences, widely analysed for evidence of a "
       "slow slip nucleation process, and a demonstration that a seismic gap can be filled "
       "only partially.",
       STANDARD),

    _e("2003 Tokachi-oki, Japan earthquake", "2003-09-25", 8.3, 27,
       "Japan", "megathrust (thrust)", "Kuril Trench subduction zone",
       ["Long-period ground motion caused sloshing that ignited oil storage tanks in Tomakomai",
        "Two deaths, low for the magnitude",
        "Well recorded by Japan's dense strong-motion networks"],
       "Made long-period ground motion a design concern for large liquid storage tanks and "
       "tall structures, since the tank fires resulted from resonance rather than from "
       "short-period shaking.",
       STANDARD),

    _e("2016 Ecuador (Muisne) earthquake", "2016-04-16", 7.8, 20,
       "Ecuador", "megathrust (thrust)", "Nazca-South America subduction zone",
       ["About 670 deaths",
        "Severe damage in coastal cities including Portoviejo and Manta",
        "Widespread collapse of non-ductile reinforced concrete buildings"],
       "Another entry in the pattern where non-ductile reinforced concrete, not magnitude, "
       "determines the death toll in a developing coastal economy.",
       IMPACT_HEAVY),

    _e("2021 Chignik, Alaska earthquake", "2021-07-29", 8.2, 32,
       "Alaska, USA", "megathrust (thrust)", "Alaska-Aleutian subduction zone",
       ["The largest US earthquake in fifty years at the time",
        "Occurred in a sparsely populated region with minimal damage",
        "Part of a sequence that included Mw 7.8 in 2020 and Mw 8.2 in 2021 along adjacent segments"],
       "A well-instrumented modern example of sequential rupture of adjacent megathrust "
       "segments, useful for testing stress-transfer models on a subduction interface.",
       BRIEF),

    _e("2009 Samoa-Tonga earthquake and tsunami", "2009-09-29", 8.1, 18,
       "Samoa / Tonga", "normal faulting at the outer rise", "Tonga Trench outer rise",
       ["An outer-rise normal-faulting event rather than a megathrust rupture",
        "Generated a tsunami that killed about 190 people in Samoa, American Samoa and Tonga",
        "Occurred in a complex doublet with a nearby megathrust rupture"],
       "Showed that outer-rise normal faulting, caused by bending of the incoming plate, "
       "can generate destructive tsunamis, and that apparently single events can be "
       "composite ruptures on different fault types.",
       STANDARD),

    # ------------------------------------------------- tsunami earthquakes
    _e("1992 Nicaragua tsunami earthquake", "1992-09-02", 7.7, 15,
       "Nicaragua", "megathrust (thrust)", "Middle America Trench",
       ["Shaking was so weak that many coastal residents did not feel it",
        "Tsunami runup reached about 10 m, killing roughly 170 people",
        "Rupture was unusually slow, lasting far longer than typical for the magnitude",
        "Radiated very little high-frequency energy"],
       "The defining modern tsunami earthquake. It proved that the absence of strong shaking "
       "is not evidence of tsunami safety, and forced tsunami warning systems to use "
       "long-period rather than short-period magnitude estimates.",
       PHYSICS_HEAVY),

    _e("2006 Java tsunami earthquake", "2006-07-17", 7.7, 20,
       "Indonesia", "megathrust (thrust)", "Sunda subduction zone",
       ["Weak shaking onshore but a tsunami with runup of several metres",
        "About 660 deaths along the southern Java coast",
        "Slow rupture propagating along the shallow plate interface"],
       "A second clear tsunami earthquake within the same decade, reinforcing that the "
       "shallow, weakly coupled updip portion of a megathrust can slip slowly and "
       "dangerously.",
       STANDARD),

    _e("1998 Papua New Guinea tsunami", "1998-07-17", 7.0, 10,
       "Papua New Guinea", "reverse", "offshore Aitape, complex plate boundary",
       ["About 2200 deaths from a tsunami far larger than the magnitude implied",
        "The tsunami is attributed largely to a submarine landslide triggered by the earthquake",
        "Runup locally reached about 15 m along a short stretch of coast",
        "Highly localised tsunami impact rather than a broad regional wave"],
       "Established submarine landslides as a first-order tsunami source. Its narrow, "
       "extreme runup could not be explained by fault displacement alone and changed how "
       "landslide tsunami hazard is assessed worldwide.",
       STANDARD),

    # ------------------------------------------------ continental collision
    _e("1950 Assam-Tibet earthquake", "1950-08-15", 8.6, 15,
       "India / Tibet", "thrust with strike-slip components", "eastern Himalayan syntaxis",
       ["The largest recorded continental earthquake not on a subduction interface",
        "Triggered enormous landslides that dammed rivers",
        "About 1500 deaths, low relative to its size because the region was sparsely populated",
        "Caused widespread changes to river courses in Assam"],
       "The upper bound on continental intraplate-collision earthquake size, and a dramatic "
       "case of earthquake-driven landscape change through landsliding and river damming.",
       STANDARD),

    _e("1934 Bihar-Nepal earthquake", "1934-01-15", 8.1, 15,
       "Nepal / India", "thrust", "Main Himalayan Thrust",
       ["About 10,000 deaths across Nepal and northern India",
        "Extensive liquefaction across the Gangetic plain, known locally as the slump belt",
        "Severe damage in Kathmandu",
        "One of the largest Himalayan earthquakes of the instrumental era"],
       "The benchmark against which the 2015 Gorkha earthquake was compared, and an early "
       "large-scale documentation of liquefaction over a broad alluvial plain.",
       STANDARD),

    _e("2011 Sikkim earthquake", "2011-09-18", 6.9, 50,
       "India / Nepal", "strike-slip", "Himalayan collision zone, within the Indian plate",
       ["About 110 deaths",
        "Deeper than typical Himalayan thrust events",
        "Strike-slip mechanism rather than thrust on the main detachment",
        "Landslides blocked mountain roads and hampered relief"],
       "Illustrates that the Himalaya hosts more than megathrust ruptures: deeper "
       "strike-slip events within the downgoing Indian plate contribute to the hazard.",
       BRIEF),

    _e("2017 Iran-Iraq border (Sarpol-e Zahab) earthquake", "2017-11-12", 7.3, 19,
       "Iran / Iraq", "thrust", "Zagros fold and thrust belt",
       ["About 630 deaths, nearly all in Iran",
        "Severe damage to recently built government housing, which became a political controversy",
        "Older traditional buildings and newer poorly supervised construction both failed"],
       "The largest Zagros earthquake in decades, and a case where the failure of recently "
       "constructed public housing highlighted enforcement rather than code content as the "
       "weak link.",
       IMPACT_HEAVY),

    _e("1990 Manjil-Rudbar, Iran earthquake", "1990-06-20", 7.4, 15,
       "Iran", "strike-slip", "Alborz mountains, northern Iran",
       ["About 40,000 deaths",
        "Struck shortly after midnight when people were asleep",
        "Widespread collapse of masonry and adobe housing in mountain villages",
        "Triggered numerous landslides"],
       "One of the deadliest Iranian earthquakes, and part of the evidence base that "
       "nighttime events in regions of heavy unreinforced masonry produce the highest "
       "fatality rates.",
       IMPACT_HEAVY),

    _e("1978 Tabas, Iran earthquake", "1978-09-16", 7.4, 15,
       "Iran", "thrust", "eastern Iran, Tabas fault",
       ["About 15,000 deaths",
        "Near-total destruction of the town of Tabas",
        "Produced one of the earliest near-fault strong-motion records from Iran"],
       "Provided an early and influential near-field accelerogram, and demonstrated the "
       "extreme vulnerability of traditional Iranian desert-town construction.",
       BRIEF),

    # -------------------------------------------- European and Mediterranean
    _e("1755 Lisbon earthquake", "1755-11-01", "~8.5", 30,
       "Portugal", "thrust", "Azores-Gibraltar plate boundary, offshore southwest Iberia",
       ["Struck on All Saints' Day when churches were full and candles were lit",
        "Destruction came from shaking, then a tsunami, then a multi-day fire",
        "Tens of thousands of deaths in Lisbon",
        "Pre-instrumental: the magnitude is an estimate from damage and tsunami records",
        "Prompted the Marques de Pombal to circulate a survey questionnaire about the shaking"],
       "Widely regarded as the birth of seismology as a science: the systematic damage "
       "survey was the first attempt to study an earthquake empirically rather than "
       "theologically, and it shaped Enlightenment philosophy.",
       STANDARD),

    _e("1980 Irpinia, Italy earthquake", "1980-11-23", 6.9, 15,
       "Italy", "normal faulting", "southern Apennine extensional belt",
       ["About 2500 deaths",
        "Complex rupture with multiple subevents over about 40 seconds",
        "Widespread collapse of stone masonry villages in the Campania region",
        "Reconstruction became a long-running case study in post-disaster governance"],
       "A multi-subevent rupture that became a standard case for source complexity, and "
       "whose troubled reconstruction is still cited in disaster-recovery policy.",
       STANDARD),

    _e("1976 Friuli, Italy earthquake", "1976-05-06", 6.5, 10,
       "Italy", "thrust", "southern Alpine front",
       ["About 1000 deaths",
        "Severe damage to historic stone masonry towns",
        "Followed by damaging aftershocks in September of the same year",
        "Reconstruction is often cited as unusually successful"],
       "The Friuli reconstruction, driven by local administration and rapid rebuilding in "
       "place, is the standard positive counterexample in post-earthquake recovery studies.",
       IMPACT_HEAVY),

    _e("1999 Athens earthquake", "1999-09-07", 6.0, 10,
       "Greece", "normal faulting", "Attica, previously unrecognised fault",
       ["About 145 deaths",
        "Occurred on a fault not previously considered a significant hazard source",
        "Damage concentrated in northern and western suburbs of Athens",
        "Prompted a rapid diplomatic thaw with Turkey after the Izmit earthquake weeks earlier"],
       "A moderate event near a capital city on an unrecognised fault, and the origin of "
       "so-called earthquake diplomacy between Greece and Turkey.",
       IMPACT_HEAVY),

    _e("2020 Aegean Sea (Izmir/Samos) earthquake", "2020-10-30", 7.0, 20,
       "Turkey / Greece", "normal faulting", "Aegean extensional province",
       ["About 120 deaths, nearly all in Izmir",
        "Damage in Izmir occurred roughly 70 km from the epicentre, on soft alluvial soils",
        "Generated a small local tsunami on Samos and the Turkish coast",
        "Collapsed mid-rise reinforced concrete buildings in the Bayrakli district"],
       "A clear modern demonstration of basin and soft-soil amplification concentrating "
       "damage in one district far from the source, echoing Mexico City in miniature.",
       STANDARD),

    _e("2020 Petrinja, Croatia earthquake", "2020-12-29", 6.4, 10,
       "Croatia", "strike-slip", "Dinarides, internal deformation zone",
       ["Seven deaths",
        "Struck during the COVID-19 pandemic, complicating shelter and evacuation",
        "Severe damage to unreinforced masonry in Petrinja and Sisak",
        "Preceded by a damaging Mw 5.4 event in Zagreb earlier the same year"],
       "A case study in compound disaster: earthquake response and pandemic control measures "
       "placed directly conflicting demands on the same population.",
       IMPACT_HEAVY),

    _e("2023 Al Haouz (Marrakesh-Safi), Morocco earthquake", "2023-09-08", 6.8, 19,
       "Morocco", "reverse (oblique)", "High Atlas mountains",
       ["About 2900 deaths",
        "Struck remote High Atlas villages built of stone and rammed earth",
        "Mountainous terrain and damaged roads delayed relief for days",
        "The strongest instrumentally recorded earthquake in Morocco"],
       "Showed that a moderate earthquake in a region with low perceived hazard, "
       "traditional earthen construction, and difficult mountain access can still produce "
       "thousands of deaths.",
       IMPACT_HEAVY),

    _e("1960 Agadir, Morocco earthquake", "1960-02-29", 5.8, 15,
       "Morocco", "strike-slip", "Atlas region, shallow local fault",
       ["Roughly 12,000 deaths from an earthquake of only moderate magnitude",
        "Extremely shallow and directly beneath the city",
        "Almost complete destruction of the city centre"],
       "The classic demonstration that a small magnitude directly beneath a vulnerable city "
       "can be deadlier than a great earthquake far away.",
       IMPACT_HEAVY),

    _e("1963 Skopje earthquake", "1963-07-26", 6.1, 10,
       "North Macedonia", "strike-slip", "Vardar zone, Balkan internal deformation",
       ["About 1070 deaths",
        "Destroyed roughly 75 percent of the city",
        "Prompted a large international reconstruction effort",
        "Rebuilding introduced modern seismic design to the region"],
       "An early case of internationally coordinated post-earthquake reconstruction, and the "
       "event that introduced modern earthquake engineering to the Balkans.",
       IMPACT_HEAVY),

    _e("2019 Durres, Albania earthquake", "2019-11-26", 6.4, 20,
       "Albania", "thrust", "Adriatic-Albanian collision front",
       ["51 deaths",
        "The deadliest earthquake in Albania in decades",
        "Collapse of mid-rise reinforced concrete apartment buildings",
        "Preceded by a smaller damaging event in September of the same year"],
       "Highlighted construction quality problems in rapidly built post-communist housing "
       "stock across the western Balkans.",
       IMPACT_HEAVY),

    # ----------------------------------------------- more North America
    _e("1971 San Fernando (Sylmar) earthquake", "1971-02-09", 6.6, 8,
       "California, USA", "thrust", "San Gabriel Mountains frontal fault",
       ["About 65 deaths, many at the Veterans Administration Hospital",
        "The nearly new Olive View Hospital was severely damaged and later demolished",
        "Freeway overpasses collapsed",
        "Produced a then-unprecedented near-field accelerogram at Pacoima Dam"],
       "The single most influential earthquake for US seismic codes. Hospital and freeway "
       "failures led directly to the Hospital Seismic Safety Act, the Alquist-Priolo fault "
       "zoning act, and a national bridge retrofit programme.",
       IMPACT_HEAVY),

    _e("1979 Imperial Valley earthquake", "1979-10-15", 6.4, 10,
       "California, USA / Mexico", "right-lateral strike-slip", "Imperial fault, Salton Trough",
       ["Recorded by a purpose-built dense near-fault accelerograph array",
        "Surface rupture along the Imperial fault, including through cultivated fields",
        "Damaged the Imperial County Services Building, a heavily instrumented structure",
        "No deaths"],
       "The El Centro differential array produced the first dense near-fault ground-motion "
       "dataset, and the instrumented building failure became a canonical structural "
       "engineering case study.",
       STANDARD),

    _e("1983 Coalinga, California earthquake", "1983-05-02", 6.2, 10,
       "California, USA", "blind thrust", "western San Joaquin Valley anticline",
       ["Occurred on a blind thrust with no surface rupture",
        "Caused surface uplift of the Anticline Ridge",
        "Severe damage to the downtown of a small agricultural town",
        "No deaths"],
       "An early recognition that growing anticlines conceal blind thrusts capable of "
       "damaging earthquakes, a decade before Northridge made the point unavoidable.",
       BRIEF),

    _e("1987 Whittier Narrows earthquake", "1987-10-01", 5.9, 14,
       "California, USA", "blind thrust", "Los Angeles basin, Puente Hills thrust system",
       ["Eight deaths",
        "Damage disproportionate to the modest magnitude",
        "Revealed the Puente Hills blind thrust beneath metropolitan Los Angeles",
        "Damaged unreinforced masonry in older neighbourhoods"],
       "Identified a blind thrust running directly beneath downtown Los Angeles, now "
       "recognised as one of the highest-consequence fault sources in the United States.",
       STANDARD),

    _e("2002 Denali, Alaska earthquake", "2002-11-03", 7.9, 5,
       "Alaska, USA", "right-lateral strike-slip", "Denali fault",
       ["Surface rupture of about 340 km",
        "Rupture propagated eastward with evidence for supershear over part of its length",
        "The Trans-Alaska Pipeline crossed the fault and survived, having been designed to slide on rails",
        "Almost no casualties because the region is sparsely populated"],
       "The best real-world validation of fault-crossing infrastructure design: the pipeline "
       "accommodated several metres of offset without rupturing, and it is also a key "
       "supershear rupture case.",
       STANDARD),

    _e("2014 South Napa, California earthquake", "2014-08-24", 6.0, 11,
       "California, USA", "right-lateral strike-slip", "West Napa fault",
       ["One death",
        "Substantial damage to unreinforced masonry in historic downtown Napa",
        "Significant losses to the wine industry from toppled barrels and tanks",
        "Notable afterslip continued on the fault for weeks"],
       "A modern well-recorded moderate event notable for prolonged afterslip and for "
       "highlighting non-structural and business-interruption losses.",
       BRIEF),

    _e("2011 Prague, Oklahoma earthquake", "2011-11-06", 5.7, 5,
       "Oklahoma, USA", "strike-slip", "intraplate, induced by wastewater injection",
       ["Among the largest earthquakes linked to wastewater disposal at the time",
        "Damaged homes and buckled a highway",
        "Occurred near long-active injection wells",
        "Part of the dramatic rise in Oklahoma seismicity after 2009"],
       "An early and heavily litigated case that helped establish the scientific and legal "
       "framework for attributing individual earthquakes to fluid injection.",
       STANDARD),

    _e("1886 Charleston, South Carolina earthquake", "1886-08-31", "~7.0", 12,
       "South Carolina, USA", "reverse", "intraplate, eastern North America",
       ["About 60 deaths",
        "Felt over an enormous area of the eastern United States",
        "Extensive liquefaction produced sand blows still visible today",
        "Pre-instrumental: magnitude estimated from damage and felt reports"],
       "The largest historical earthquake in the eastern United States, and the primary "
       "evidence that damaging intraplate earthquakes occur along the Atlantic seaboard.",
       STANDARD),

    _e("1811-1812 New Madrid earthquake sequence", "1811-12-16", "~7.5", 10,
       "Missouri, USA", "strike-slip and reverse", "intraplate, Reelfoot Rift",
       ["A sequence of at least three very large events over about two months",
        "Felt across most of the eastern United States",
        "Created Reelfoot Lake by tectonic subsidence",
        "Caused extensive liquefaction across the Mississippi embayment",
        "Pre-instrumental: magnitudes are estimated and actively debated"],
       "The defining intraplate earthquake sequence of North America. Its recurrence "
       "interval and magnitude remain contested, and that uncertainty dominates seismic "
       "hazard estimates for the central United States.",
       STANDARD),

    _e("1985 Nahanni, Canada earthquake", "1985-12-23", 6.8, 8,
       "Northwest Territories, Canada", "thrust", "intraplate, Mackenzie Mountains",
       ["Recorded peak ground accelerations exceeding 1 g, exceptionally high for the time",
        "Occurred in a remote, essentially unpopulated region",
        "One of very few strong-motion recordings from a stable continental region"],
       "Its extreme recorded accelerations forced attention to the possibility of very high "
       "short-period ground motion in stable continental crust.",
       BRIEF),

    # ---------------------------------------------------- more Japan / Asia
    _e("2024 Noto Peninsula earthquake", "2024-01-01", 7.5, 10,
       "Japan", "reverse", "Noto Peninsula, Sea of Japan coast",
       ["Struck on New Year's Day",
        "Over 200 deaths",
        "Caused coastal uplift of several metres, stranding harbours above sea level",
        "Preceded by a multi-year swarm attributed to deep fluid migration",
        "Generated a local tsunami and triggered widespread landslides"],
       "A rare case where a years-long earthquake swarm, interpreted as fluid-driven, "
       "preceded a major rupture, and a striking example of permanent coseismic coastal "
       "uplift altering harbour infrastructure.",
       STANDARD),

    _e("2024 Hualien, Taiwan earthquake", "2024-04-03", 7.4, 35,
       "Taiwan", "reverse", "Taiwan orogen, Longitudinal Valley region",
       ["The strongest Taiwan earthquake since 1999",
        "Fewer than 20 deaths, many from rockfalls rather than building collapse",
        "Extensive landslides in the Taroko Gorge area",
        "Building performance was generally good after decades of code improvement"],
       "A direct measure of the progress made since Chi-Chi in 1999: a comparable magnitude "
       "event produced roughly two orders of magnitude fewer deaths.",
       STANDARD),

    _e("2018 Hokkaido Eastern Iburi earthquake", "2018-09-06", 6.6, 35,
       "Japan", "reverse", "Hokkaido, intraplate crustal fault",
       ["41 deaths, most caused by landslides rather than shaking",
        "Thousands of shallow landslides stripped hillsides of volcanic ash soil",
        "Caused a blackout across the whole of Hokkaido when a major power station failed"],
       "Notable for two cascading failures: pervasive landsliding in weak volcanic soils, "
       "and the first island-wide blackout in Japan, which exposed the fragility of "
       "concentrated power generation.",
       IMPACT_HEAVY),

    _e("2007 Niigata Chuetsu-oki earthquake", "2007-07-16", 6.6, 10,
       "Japan", "reverse", "Sea of Japan coast, offshore Niigata",
       ["Ground motion at the Kashiwazaki-Kariwa nuclear plant exceeded its design basis",
        "The plant shut down safely but remained offline for years",
        "11 deaths",
        "A small fire occurred at an electrical transformer at the plant"],
       "The first case of a nuclear plant experiencing shaking well beyond its design basis. "
       "It triggered global re-evaluation of nuclear seismic design margins, four years "
       "before Fukushima.",
       STANDARD),

    _e("2004 Niigata Chuetsu earthquake", "2004-10-23", 6.6, 13,
       "Japan", "reverse", "Niigata, inland fold-and-thrust belt",
       ["About 68 deaths",
        "Derailed a Shinkansen bullet train, the first such derailment in service",
        "Extensive landslides in mountainous terrain isolated villages",
        "Numerous strong aftershocks"],
       "The Shinkansen derailment ended the system's perfect safety record and drove "
       "development of early-warning-triggered automatic braking.",
       STANDARD),

    _e("2008 Iwate-Miyagi Nairiku earthquake", "2008-06-13", 6.9, 8,
       "Japan", "reverse", "northeast Honshu, inland crustal fault",
       ["Recorded among the highest peak ground accelerations ever measured, exceeding 4 g at one station",
        "About 23 deaths, low given the extreme recorded motion",
        "Triggered very large landslides in mountainous terrain"],
       "Its extraordinary recorded accelerations, with comparatively modest damage, sharpened "
       "the understanding that peak acceleration alone is a poor predictor of structural "
       "damage compared with velocity and duration.",
       PHYSICS_HEAVY),

    _e("2006 Yogyakarta, Indonesia earthquake", "2006-05-27", 6.3, 12,
       "Indonesia", "strike-slip", "Java, crustal fault near Mount Merapi",
       ["About 5700 deaths",
        "Very shallow and close to a densely populated area",
        "Widespread collapse of unreinforced brick housing",
        "Occurred while Mount Merapi was in an active eruptive phase"],
       "A moderate-magnitude, high-fatality event driven by shallow depth and non-engineered "
       "masonry housing, and a case of overlapping volcanic and seismic emergencies.",
       IMPACT_HEAVY),

    _e("2013 Bohol, Philippines earthquake", "2013-10-15", 7.1, 12,
       "Philippines", "reverse", "previously unmapped North Bohol fault",
       ["About 220 deaths",
        "Produced a distinct surface rupture scarp several metres high",
        "Destroyed a number of centuries-old stone churches",
        "Occurred on a fault not previously mapped"],
       "Another unmapped-fault surprise, and a significant loss of built cultural heritage "
       "that shaped subsequent conservation and retrofit policy in the Philippines.",
       STANDARD),

    _e("2018 Lombok, Indonesia earthquake sequence", "2018-08-05", 6.9, 31,
       "Indonesia", "reverse", "Flores back-arc thrust",
       ["A sequence of damaging events through July and August 2018",
        "About 560 deaths in total",
        "Caused widespread damage to non-engineered housing across northern Lombok",
        "Coseismic uplift of the island was measured geodetically"],
       "A back-arc thrust sequence rather than a subduction megathrust, showing that the "
       "hazard behind an arc can rival that in front of it.",
       IMPACT_HEAVY),

    _e("1975 Haicheng, China earthquake", "1975-02-04", 7.0, 12,
       "China", "strike-slip", "Liaoning, intraplate northeastern China",
       ["Preceded by a pronounced foreshock sequence and reported anomalies",
        "An evacuation was ordered shortly before the mainshock",
        "The evacuation is credited with saving many lives",
        "Deaths still numbered in the thousands"],
       "The most cited apparent success in earthquake prediction. Its reputation created "
       "expectations that the Tangshan earthquake shattered eighteen months later, and it "
       "remains debated how much was foresight and how much was fortunate timing.",
       STANDARD),

    _e("2013 Lushan, China earthquake", "2013-04-20", 6.6, 14,
       "China", "thrust", "southern Longmenshan fault zone",
       ["About 200 deaths",
        "Occurred on the southern segment of the fault system that produced Wenchuan in 2008",
        "Triggered landslides in steep terrain",
        "Buildings rebuilt after 2008 generally performed well"],
       "Demonstrated both stress transfer along the Longmenshan system and the measurable "
       "benefit of post-Wenchuan reconstruction standards.",
       STANDARD),

    _e("2022 Khost, Afghanistan earthquake", "2022-06-22", 6.0, 10,
       "Afghanistan", "strike-slip", "Sulaiman range, Afghanistan-Pakistan border region",
       ["About 1000 deaths from a moderate magnitude event",
        "Struck a remote region of stone and mud-brick housing",
        "Occurred at night",
        "Relief was severely constrained by the political situation and terrain"],
       "A stark modern reminder that Mw 6 events remain lethal where housing is "
       "non-engineered and emergency response capacity is limited.",
       IMPACT_HEAVY),

    _e("2021 Nippes, Haiti earthquake", "2021-08-14", 7.2, 10,
       "Haiti", "oblique reverse and strike-slip", "Enriquillo-Plantain Garden fault zone",
       ["About 2200 deaths",
        "Struck a different segment of the same fault system as the 2010 earthquake",
        "Tropical Storm Grace arrived days later, hampering relief",
        "Fewer deaths than 2010 despite a larger magnitude, partly because it struck a less dense area"],
       "A compound disaster where a tropical storm followed the earthquake within days, and "
       "evidence that the 2010 rupture had loaded adjacent segments of the fault system.",
       IMPACT_HEAVY),

    # ------------------------------------------------ induced and unusual
    _e("2017 Pohang, South Korea earthquake", "2017-11-15", 5.5, 7,
       "South Korea", "strike-slip", "induced by enhanced geothermal system stimulation",
       ["The second largest instrumentally recorded earthquake in South Korea",
        "About 135 injuries and extensive damage, but no deaths",
        "A government investigation concluded it was induced by nearby geothermal stimulation",
        "The geothermal project was permanently shut down"],
       "The most consequential induced earthquake from enhanced geothermal systems, and the "
       "clearest case of an official inquiry formally attributing a damaging earthquake to a "
       "specific energy project.",
       STANDARD),

    _e("2006 Basel, Switzerland induced earthquake", "2006-12-08", 3.4, 5,
       "Switzerland", "strike-slip", "induced by enhanced geothermal system stimulation",
       ["Induced during hydraulic stimulation of a deep geothermal reservoir beneath the city",
        "Too small to cause structural damage but widely felt and alarming",
        "Stimulation was halted and the project was ultimately abandoned",
        "Generated substantial insurance claims for minor damage"],
       "The event that ended deep geothermal development beneath European cities for years, "
       "and the origin of traffic-light protocols governing injection operations.",
       BRIEF),

    _e("1967 Koyna, India reservoir-induced earthquake", "1967-12-10", 6.3, 10,
       "India", "strike-slip", "induced by reservoir impoundment, Deccan Traps",
       ["About 180 deaths",
        "Followed impoundment of the Koyna reservoir several years earlier",
        "Occurred in a region previously considered aseismic",
        "Seismicity in the area has continued for decades"],
       "The largest and most widely accepted case of reservoir-induced seismicity, and the "
       "reason large dam projects now include seismic monitoring as standard.",
       STANDARD),

    _e("2015 Bonin Islands deep earthquake", "2015-05-30", 7.8, 680,
       "Japan (Bonin Islands)", "within the deep slab", "deep subducted Pacific slab",
       ["Among the deepest large earthquakes ever recorded, near 680 km",
        "Felt across all of Japan, an unusual footprint for a deep event",
        "Occurred deeper than most of the well-defined Wadati-Benioff zone in the region",
        "Caused no significant damage"],
       "Its depth near the base of the transition zone tests the limits of the "
       "transformational-faulting mechanism, since metastable olivine should be exhausted "
       "at those conditions.",
       PHYSICS_HEAVY),

    _e("1970 Colombia deep-focus earthquake", "1970-07-31", 8.0, 645,
       "Colombia", "within the deep slab", "deep subducted Nazca slab",
       ["A very large earthquake at about 645 km depth",
        "Caused essentially no damage despite its magnitude",
        "Felt over an extremely wide area of South America"],
       "One of the small handful of Mw 8 class deep-focus earthquakes, and part of the "
       "limited dataset constraining rupture physics at transition-zone depths.",
       BRIEF),

    _e("2018 Kilauea summit collapse earthquake", "2018-05-04", 6.9, 6,
       "Hawaii, USA", "thrust on the volcano's basal detachment", "Kilauea volcano flank",
       ["The largest Hawaiian earthquake since 1975",
        "Accompanied the 2018 Kilauea eruption and caldera collapse sequence",
        "Caused seaward slumping of the volcano's south flank",
        "Generated a small local tsunami"],
       "Illustrates volcano-tectonic coupling: magma withdrawal and flank instability "
       "produced a large earthquake on the volcano's basal detachment rather than on a "
       "conventional tectonic fault.",
       STANDARD),

    _e("1975 Kalapana, Hawaii earthquake", "1975-11-29", 7.7, 9,
       "Hawaii, USA", "thrust on the volcano's basal detachment", "Kilauea south flank",
       ["Two deaths from the resulting local tsunami",
        "Caused several metres of subsidence along the Kilauea coast",
        "Triggered a brief eruption at Kilauea's summit"],
       "The largest Hawaiian earthquake of the twentieth century and the benchmark for "
       "volcano flank collapse hazard, including its capacity to generate local tsunamis.",
       BRIEF),

    _e("1995 Neftegorsk, Sakhalin earthquake", "1995-05-27", 7.0, 11,
       "Russia (Sakhalin)", "strike-slip", "Sakhalin, intraplate deformation zone",
       ["About 2000 deaths in a town of roughly 3000 people",
        "Nearly complete collapse of five-storey prefabricated concrete apartment blocks",
        "The town was never rebuilt",
        "Struck shortly after 1 a.m. local time"],
       "The highest proportional death toll of any modern earthquake in a single settlement, "
       "and a definitive indictment of unreinforced prefabricated panel construction.",
       IMPACT_HEAVY),

    _e("1980 El Asnam, Algeria earthquake", "1980-10-10", 7.1, 10,
       "Algeria", "thrust", "Tell Atlas, northern Algeria",
       ["About 2600 deaths",
        "Produced a prominent surface rupture with metres of vertical offset",
        "The same town had been destroyed by an earthquake in 1954",
        "Extensive collapse of masonry and early reinforced concrete buildings"],
       "A well-documented thrust surface rupture, and a case where the same settlement was "
       "destroyed twice within a generation, raising hard questions about rebuilding in place.",
       STANDARD),

    _e("1931 Hawke's Bay (Napier), New Zealand earthquake", "1931-02-03", 7.8, 20,
       "New Zealand", "reverse", "Hikurangi margin, upper plate faulting",
       ["256 deaths, the deadliest earthquake in New Zealand history",
        "Uplifted about 40 square kilometres of seabed, creating new land at Napier",
        "Fire destroyed much of the central business district after the shaking",
        "Napier was rebuilt in the Art Deco style of the period"],
       "The origin of New Zealand's modern building code and earthquake engineering "
       "profession, and a dramatic example of large permanent coseismic uplift.",
       STANDARD),

    _e("1939 Erzincan, Turkey earthquake", "1939-12-26", 7.8, 20,
       "Turkey", "right-lateral strike-slip", "North Anatolian Fault",
       ["About 33,000 deaths",
        "Struck in winter, compounding exposure and hampering relief",
        "Produced surface rupture of roughly 350 km",
        "The first in the twentieth-century westward migrating sequence along the fault"],
       "The starting point of the celebrated westward progression of large ruptures along "
       "the North Anatolian Fault, one of the clearest examples of sequential stress "
       "transfer along a major fault system.",
       STANDARD),

    _e("1968 Dasht-e Bayaz, Iran earthquake", "1968-08-31", 7.1, 13,
       "Iran", "left-lateral strike-slip", "eastern Iran fault system",
       ["About 12,000 deaths",
        "Produced about 80 km of surface rupture with several metres of offset",
        "Near-total destruction of villages built of mud brick",
        "Studied in detail by early field geologists documenting surface faulting"],
       "One of the best-documented early strike-slip surface ruptures, and a foundational "
       "field dataset for relating rupture length and offset to magnitude.",
       BRIEF),

    _e("2011 Van, Turkey earthquake", "2011-10-23", 7.1, 18,
       "Turkey", "thrust", "eastern Anatolia, collision zone",
       ["About 600 deaths",
        "A second damaging event struck the same area weeks later, collapsing a hotel",
        "Winter conditions complicated sheltering the displaced population",
        "Widespread collapse of poorly built reinforced concrete and masonry"],
       "A reminder that eastern Anatolia's thrust and collision structures, not only the "
       "North Anatolian Fault, pose major hazard, and that aftershocks can destroy buildings "
       "already weakened by the mainshock.",
       IMPACT_HEAVY),

    _e("2001 Nisqually, Washington earthquake", "2001-02-28", 6.8, 52,
       "Washington, USA", "normal faulting within the slab", "intraslab, subducted Juan de Fuca plate",
       ["No deaths directly from the shaking",
        "Its depth of about 52 km reduced surface shaking intensity",
        "Caused significant damage to unreinforced masonry in Seattle's Pioneer Square",
        "Similar in character to intraslab events in 1949 and 1965"],
       "A well-recorded Cascadia intraslab event that demonstrates how depth attenuates "
       "damage, and that this earthquake type, rather than the megathrust, dominates "
       "frequent shaking in the Puget Sound region.",
       STANDARD),

    _e("2016 Kaikoura, New Zealand earthquake", "2016-11-13", 7.8, 15,
       "New Zealand", "complex multi-fault, reverse and strike-slip", "Marlborough fault system, Hikurangi margin",
       ["Ruptured more than twenty separate faults in a single event",
        "Two deaths",
        "Coastal uplift of up to about 6 m exposed seafloor at Kaikoura",
        "Massive landslides severed the main highway and rail line for months",
        "Rupture propagated across large gaps between mapped faults"],
       "The most complex multi-fault rupture ever documented. It demolished the assumption "
       "that ruptures stop at fault segment boundaries or step-overs beyond a few kilometres, "
       "forcing revision of how maximum magnitudes are estimated in hazard models.",
       PHYSICS_HEAVY),
]


def total_pairs() -> int:
    return sum(sum(e["mix"].values()) for e in EARTHQUAKES)


def summary() -> str:
    from collections import Counter
    kinds: Counter = Counter()
    for e in EARTHQUAKES:
        for k, v in e["mix"].items():
            kinds[k] += v
    regions = Counter(e["region"].split(" / ")[0].split(",")[0] for e in EARTHQUAKES)
    mags = [e["mw"] for e in EARTHQUAKES if isinstance(e["mw"], (int, float))]
    lines = [f"{len(EARTHQUAKES)} earthquakes -> {total_pairs()} Q&A pairs requested", ""]
    lines.append(f"magnitude range: Mw {min(mags):.1f} - {max(mags):.1f}")
    depths = [e["depth_km"] for e in EARTHQUAKES]
    lines.append(f"depth range:     {min(depths)} - {max(depths)} km")
    lines += ["", "by question type:"]
    lines += [f"  {k:<14} {n:>3} pairs" for k, n in sorted(kinds.items())]
    lines += ["", "by country/region (top 8):"]
    lines += [f"  {r:<16} {n}" for r, n in regions.most_common(8)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
