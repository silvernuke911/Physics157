import numpy as np
from modules import constants

def compute_orbital_elements(r_i, v_i, G, M):
    if len(r_i)==2:                         # Creating 3d vectors if vectors are 2d
        r_i=np.append(r_i,0)
        v_i=np.append(v_i,0)
    mu = G * M                              # Gravitational parameter (mu = GM)
    h = np.cross(r_i, v_i)                  # Specific angular momentum vector (h = r x v)
    h_mag = np.linalg.norm(h)               # Magnitude of specific angular momentum
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))   # Eccentricity vector (e = ((v x h) / mu) - (r / |r|))
    e_mag = np.linalg.norm(e)               # Magnitude of eccentricity vector
    r_mag = np.linalg.norm(r_i)             # Semi-major axis (a)
    v_mag = np.linalg.norm(v_i)
    a = 1 / ((2 / r_mag) - (v_mag ** 2 / mu))
    T_p = 2 * np.pi * np.sqrt( a**3 / mu)   # Orbital period
    i = np.arccos(h[2] / h_mag)             # Inclination (i)
    k = np.array([0, 0, 1])                 # Node vector (n = k x h)
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)               # Magnitude of node vector
    if n_mag != 0:                          # Right ascension of the ascending node (RAAN, Ω)
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0
    if n_mag != 0 and e_mag != 0:           # Argument of periapsis (ω)
        omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
        if e[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0
    if e_mag != 0:                          # True anomaly (ν)
        nu = np.arccos(np.dot(e, r_i) / (e_mag * r_mag))
        if np.dot(r_i, v_i) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0
    orbital_elements = {                    # Return orbital elements as a dictionary
        'semi_major_axis': a,
        'eccentricity': e_mag,
        'inclination': np.degrees(i),
        'LAN': np.degrees(Omega),
        'argument_of_periapsis': np.degrees(omega),
        'true_anomaly': np.degrees(nu),
        'orbital period': T_p
    }
    return orbital_elements

def state_vectors(a, e, i, Omega, omega, M0, t, mu, t0=0, units="AU-day"):
    """
    Computes the position and velocity vectors at time t from Keplerian orbital elements.

    Parameters
    ----------
    a : float
        Semi-major axis [AU or m, depending on unit system].
    e : float
        Eccentricity.
    i : float
        Inclination [degrees].
    Omega : float
        Longitude of the ascending node [degrees].
    omega : float
        Argument of periapsis [degrees].
    M0 : float
        Mean anomaly at epoch t0 [degrees].
    t : float
        Time since epoch t0 [days or seconds depending on unit system].
    mu : float
        Gravitational parameter G*M [AU³/day² or m³/s²].
    t0 : float, optional
        Epoch time of given M0.
    units : str, optional
        Unit system to return. Options: "AU-day" or "m-s".

    Returns
    -------
    r_vec : ndarray
        Position vector at time t [AU or m].
    v_vec : ndarray
        Velocity vector at time t [AU/day or m/s].
    """
    # Conversion constants
    AU_in_m = 1.495978707e11  # meters
    day_in_s = 86400          # seconds

    # Convert angles from degrees to radians
    i_rad = np.radians(i)
    Omega_rad = np.radians(Omega)
    omega_rad = np.radians(omega)
    M0_rad = np.radians(M0)

    # Mean motion
    n = np.sqrt(mu / a**3)

    # Mean anomaly at time t
    M = M0_rad + n * (t - t0)

    # Solve Kepler's Equation using Newton-Raphson
    def kepler(E): return E - e * np.sin(E) - M
    def kepler_prime(E): return 1 - e * np.cos(E)

    E = M
    for _ in range(100):
        delta = kepler(E) / kepler_prime(E)
        E -= delta
        if abs(delta) < 1e-9:
            break

    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))

    r_mag = a * (1 - e * np.cos(E))

    x_orb = r_mag * np.cos(nu)
    y_orb = r_mag * np.sin(nu)

    # Velocity in orbital plane
    v_r = np.sqrt(mu / a) / (1 - e * np.cos(E)) * (-np.sin(E))
    v_theta = np.sqrt(mu / a) / (1 - e * np.cos(E)) * (np.sqrt(1 - e**2) * np.cos(E))

    vx_orb = v_r * np.cos(nu) - v_theta * np.sin(nu)
    vy_orb = v_r * np.sin(nu) + v_theta * np.cos(nu)

    # Rotation matrix from perifocal to ECI
    cos_O, sin_O = np.cos(Omega_rad), np.sin(Omega_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_w, sin_w = np.cos(omega_rad), np.sin(omega_rad)

    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i,                          cos_w * sin_i,                          cos_i]
    ])

    r_orb = np.array([x_orb, y_orb, 0])
    v_orb = np.array([vx_orb, vy_orb, 0])

    r_vec = R @ r_orb
    v_vec = R @ v_orb

    # Convert units if needed
    if units == "m-s":
        r_vec *= AU_in_m
        v_vec *= AU_in_m / day_in_s  # AU/day → m/s

    return r_vec, v_vec


def body_state_vectors(body_name):
    return state_vectors(
        a     = constants.orbital_parameters[body_name]["a"],
        e     = constants.orbital_parameters[body_name]["e"],
        i     = constants.orbital_parameters[body_name]["i"],  
        Omega = constants.orbital_parameters[body_name]["Omega"],       
        omega = constants.orbital_parameters[body_name]["omega"],       
        M0    = constants.orbital_parameters[body_name]["M0"],        
        t     = 0,
        mu    = constants.G_ausmday ,                     # AU^3/day^2
        t0    = 0                                    # J2000 epoch JD
    )

def orbit_3d(a, e, i, Omega, omega, theta=None, size = 2000):
    """
    Returns 3D position vectors r = [x, y, z] for an orbit.
    
    Parameters:
    - a : float, semi-major axis
    - e : float, eccentricity
    - i : float, inclination (degrees)
    - Omega : float, longitude of ascending node (degrees)
    - omega : float, argument of periapsis (degrees)
    - theta : np.ndarray, true anomaly array (radians), optional
    
    Returns:
    - r : (N, 3) ndarray of 3D coordinates [x, y, z]
    """
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, size)

    # Convert angles to radians
    i_rad = np.radians(i)
    Omega_rad = np.radians(Omega)
    omega_rad = np.radians(omega)

    # Orbital radius at each true anomaly
    r_orbit = a * (1 - e**2) / (1 + e * np.cos(theta))

    # Coordinates in orbital plane
    x_orb = r_orbit * np.cos(theta)
    y_orb = r_orbit * np.sin(theta)
    z_orb = np.zeros_like(x_orb)

    # Rotate: perifocal → geocentric equatorial frame
    # Rotation matrix: Rz(-Omega) * Rx(-i) * Rz(-omega)
    cos_O, sin_O = np.cos(Omega_rad), np.sin(Omega_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_w, sin_w = np.cos(omega_rad), np.sin(omega_rad)

    R = np.array([
        [
            cos_O * cos_w - sin_O * sin_w * cos_i,
            -cos_O * sin_w - sin_O * cos_w * cos_i,
            sin_O * sin_i
        ],
        [
            sin_O * cos_w + cos_O * sin_w * cos_i,
            -sin_O * sin_w + cos_O * cos_w * cos_i,
            -cos_O * sin_i
        ],
        [
            sin_w * sin_i,
            cos_w * sin_i,
            cos_i
        ]
    ])

    # Stack and rotate all points
    r_orbital = np.vstack((x_orb, y_orb, z_orb))  # shape (3, N)
    r_3d = R @ r_orbital                         # shape (3, N)
    
    return r_3d.T  # shape (N, 3)

def body_orbit3d(body_name, size):
    return orbit_3d(
        a     = constants.orbital_parameters[body_name]["a"],
        e     = constants.orbital_parameters[body_name]["e"],
        i     = constants.orbital_parameters[body_name]["i"],  
        Omega = constants.orbital_parameters[body_name]["Omega"],       
        omega = constants.orbital_parameters[body_name]["omega"],   
        size  = size
    )