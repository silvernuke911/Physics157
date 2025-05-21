# Gravitational constants
G_mks = 6.67430e-11  # m^3 kg^-1 s^-2
G_aukgday = 2.959122082855911e-4  # AU^3 / (kg * day^2), optional alt unit

# Distance unit
AU = 149597870700  # meters

# Celestial body properties
properties = {
    "Sun": {
        "mass": 1.9891e30,
        "radius": 6.9634e8,
        "GM": 1.32712440018e20
    },
    "Earth": {
        "mass": 5.9722e24,
        "radius": 6.371e6,
        "GM": 3.986004418e14
    },
    "Jupiter": {
        "mass": 1.8982e27,
        "radius": 6.9911e7,
        "GM": 1.26686534e17
    },
    "Saturn": {
        "mass": 5.6834e26,
        "radius": 5.8232e7,
        "GM": 3.7931187e16
    },
    "Mercury": {
        "mass": 3.3011e23,
        "radius": 2.4397e6,
        "GM": 2.2032e13
    },
    "Venus": {
        "mass": 4.8675e24,
        "radius": 6.0518e6,
        "GM": 3.24859e14
    },
    "Mars": {
        "mass": 6.4171e23,
        "radius": 3.3895e6,
        "GM": 4.282837e13
    },
    "Neptune": {
        "mass": 1.02413e26,
        "radius": 2.4622e7,
        "GM": 6.836529e15
    },
    "Uranus": {
        "mass": 8.6810e25,
        "radius": 2.5362e7,
        "GM": 5.793939e15
    },
    "Apophis": {
        "mass": 6.1e10,
        "radius": None,
        "GM": None
    }
}

# Orbital parameters (Epoch J2000 unless noted)
orbital_parameters = {
    "Mercury": {
        "a": 0.387098,
        "e": 0.205630,
        "i": 7.00487,
        "Omega": 48.33167,
        "omega": 29.12478,
        "M0": 174.79588
    },
    "Venus": {
        "a": 0.723332,
        "e": 0.006773,
        "i": 3.39471,
        "Omega": 76.68069,
        "omega": 54.85229,
        "M0": 50.41611
    },
    "Earth": {
        "a": 1.000000,
        "e": 0.016710,
        "i": 0.00005,
        "Omega": -11.26064,
        "omega": 114.20783,
        "M0": 358.617
    },
    "Mars": {
        "a": 1.523679,
        "e": 0.093400,
        "i": 1.85061,
        "Omega": 49.57854,
        "omega": 286.46230,
        "M0": 19.41248
    },
    "Jupiter": {
        "a": 5.204267,
        "e": 0.048775,
        "i": 1.30530,
        "Omega": 100.55615,
        "omega": 273.867,
        "M0": 20.0202
    },
    "Saturn": {
        "a": 9.582017,
        "e": 0.055723,
        "i": 2.48446,
        "Omega": 113.71504,
        "omega": 339.392,
        "M0": 317.0207
    },
    "Uranus": {
        "a": 19.18926,
        "e": 0.044405,
        "i": 0.76986,
        "Omega": 74.22988,
        "omega": 96.998857,
        "M0": 142.2386
    },
    "Neptune": {
        "a": 30.069922,
        "e": 0.011214,
        "i": 1.76917,
        "Omega": 131.72169,
        "omega": 272.8461,
        "M0": 256.228
    },
    "Apophis": {
        "a": 0.9224,
        "e": 0.1911,
        "i": 3.341,
        "Omega": 203.9,
        "omega": 126.7,
        "M0": 90.28,
        "M0_J2000": 333.26,
        "epoch_jd": 2460800.5,
        "P": 323.6,
        "mean_motion": 1.112,
        "avg_speed": 30.73  # km/s
    }
}
