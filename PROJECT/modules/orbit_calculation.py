import numpy as np

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