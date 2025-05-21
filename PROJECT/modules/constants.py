# Gravitational constants
G_mks = 6.67430e-11  # m^3 kg^-1 s^-2
G_ausmday = 2.959122082855911e-4  # AU^3 / (kg * day^2), optional alt unit

# Distance unit
AU = 149597870700  # meters


# Julian Dates
def JD(year, month=1, day=0, hour=0, minute=0, second=0):
    """
    Compute the Julian Date from a Gregorian date and time.
    
    Parameters:
        year (int): Year
        month (int): Month (default 1)
        day (int): Day (default 0)
        hour (int): Hour (default 0)
        minute (int): Minute (default 0)
        second (int): Second (default 0)
    
    Returns:
        float: Julian Date
    """
    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4

    JD_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    JD_frac = (hour + minute / 60 + second / 3600) / 24

    return JD_day + JD_frac

J2000 = 2451545.0   # January 1, 2000 at 12:00 TT
J2005 = 2453371.0   # January 1, 2005 at 12:00 TT
J2010 = 2455197.0   # January 1, 2010 at 12:00 TT
J2015 = 2457023.0   # January 1, 2015 at 12:00 TT
J2020 = 2458849.0   # January 1, 2020 at 12:00 TT

# Celestial body properties
bodies = {
    "Sun": {
        "mass": 1.9891e30,
        "radius": 6.9634e8,
        "GM": 1.32712440018e20,
        "GM_au": 392.4987725517698,
        "mass_solar": 1.0
    },
    "Earth": {
        "mass": 5.9722e24,
        "radius": 6.371e6,
        "GM": 3.986004418e14,
        "GM_au": 0.11759853917415852,
        "mass_solar": 3.0026861710270094e-06
    },
    "Jupiter": {
        "mass": 1.8982e27,
        "radius": 6.9911e7,
        "GM": 1.26686534e17,
        "GM_au": 37.47202427742101,
        "mass_solar": 0.0009543187293775984
    },
    "Saturn": {
        "mass": 5.6834e26,
        "radius": 5.8232e7,
        "GM": 3.7931187e16,
        "GM_au": 11.22221403270052,
        "mass_solar": 0.00028571619681911624
    },
    "Mercury": {
        "mass": 3.3011e23,
        "radius": 2.4397e6,
        "GM": 2.2032e13,
        "GM_au": 0.006520215573026429,
        "mass_solar": 1.6595871061304174e-07
    },
    "Venus": {
        "mass": 4.8675e24,
        "radius": 6.0518e6,
        "GM": 3.24859e14,
        "GM_au": 0.09579924125890915,
        "mass_solar": 2.447838339664124e-06
    },
    "Mars": {
        "mass": 6.4171e23,
        "radius": 3.3895e6,
        "GM": 4.282837e13,
        "GM_au": 0.012667035525672004,
        "mass_solar": 3.2264186280032343e-07
    },
    "Neptune": {
        "mass": 1.02413e26,
        "radius": 2.4622e7,
        "GM": 6.836529e15,
        "GM_au": 2.0223823606088865,
        "mass_solar": 5.149866177559703e-05
    },
    "Uranus": {
        "mass": 8.6810e25,
        "radius": 2.5362e7,
        "GM": 5.793939e15,
        "GM_au": 1.7141849021842167,
        "mass_solar": 4.364350452528864e-05
    },
    "Apophis": {
        "mass": 6.1e10,
        "radius": None,
        "GM": None,
        "GM_au": None,
        "mass_solar": 3.067647296964195e-20
    }
}


# Orbital parameters (Epoch J2000 unless noted)
orbital_parameters = {
    "Mercury": {
        "a": 0.387098,             # Semi-major axis [AU]
        "e": 0.205630,             # Eccentricity
        "i": 7.00487,              # Inclination [deg]
        "Omega": 48.33167,         # Longitude of ascending node [deg]
        "omega": 29.12478,         # Argument of perihelion [deg]
        "M0": 174.79588            # Mean anomaly at epoch [deg]
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
        "i": 3.341,                # Inclination [deg]
        "Omega": 203.9,            # Longitude of ascending node [deg]
        "omega": 126.7,            # Argument of perihelion [deg]
        "M0": 333.26,        # Mean anomaly at J2000.0 [deg]
        "P": 323.6,                # Orbital period [days]
        "mean_motion": 1.112,      # Mean motion [deg/day]
        "avg_speed": 30.73         # Average orbital speed [km/s]
    }
}


