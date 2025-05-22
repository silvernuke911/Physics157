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

def grav_acc_nbody(t, R, V, masses, G):
    """
    Compute gravitational accelerations on each body due to every other body.

    Parameters
    ----------
    t : float
        Time (unused, kept for API consistency)
    R : ndarray
        Positions of shape (N, D)
    V : ndarray
        Velocities of shape (N, D) (unused)
    masses : ndarray
        Array of masses of shape (N,)
    G : float
        Gravitational constant

    Returns
    -------
    A : ndarray
        Accelerations of shape (N, D)
    """
    N, D = R.shape
    A = np.zeros((N, D))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_ij = R[j] - R[i]
            dist_sq = np.dot(r_ij, r_ij)
            if dist_sq != 0:
                A[i] += G * masses[j] * r_ij / (dist_sq * np.sqrt(dist_sq))
    
    return A
