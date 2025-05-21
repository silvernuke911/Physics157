import numpy as np
def grav_acc(t, r, v, m, G):
    """
    Compute gravitational acceleration at position r due to a central mass m.

    Parameters
    ----------
    t : float
        Current time (not used here, included for API compatibility)
    r : ndarray
        Position vector [length 3]
    v : ndarray
        Velocity vector [length 3] (not used here, included for API compatibility)
    m : float
        Mass of the central object
    G : float
        Gravitational constant

    Returns
    -------
    a : ndarray
        Acceleration vector at position r
    """
    mu = G * m
    r_mag_sq = np.dot(r, r)
    if r_mag_sq == 0:
        return np.zeros_like(r)
    return -mu * r / (r_mag_sq * np.sqrt(r_mag_sq))
