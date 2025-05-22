import numpy as np
import time
from modules import progress

def kinematic_rk4(t, f, r0, v0, progressbar = False):
    """
    Simulates the motion of an object using the 4th-order Runge-Kutta (RK4) method
    for systems described by position, velocity, and a time-dependent acceleration function.

    Parameters
    ----------
    t : ndarray
        1D array of time points at which to compute the solution. Assumes uniform spacing.
    f : callable
        Function of the form `f(t, r, v)` returning the acceleration (a vector) at time `t`,
        position `r`, and velocity `v`.
    r0 : ndarray
        Initial position vector.
    v0 : ndarray
        Initial velocity vector.

    Returns
    -------
    r : ndarray
        Array of positions at each time step. Shape: (len(t), len(r0)).
    v : ndarray
        Array of velocities at each time step. Shape: (len(t), len(v0)).
    a : ndarray
        Array of accelerations at each time step. Shape: (len(t), len(v0)).
    """
    if progressbar:
        time_start = time.time()
    dt = t[1] - t[0]  # Assuming uniform time steps
    half_dt = dt / 2
    dim = len(r0)

    r = np.zeros((len(t), dim))
    v = np.zeros((len(t), dim))
    a = np.zeros((len(t), dim))

    r[0] = r0
    v[0] = v0
    a[0] = f(t[0], r0, v0)

    for i in range(len(t) - 1):
        t_i = t[i]
        r_i = r[i]
        v_i = v[i]

        k1_v = f(t_i, r_i, v_i)
        k2_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k1_v * half_dt)
        k3_v = f(t_i + half_dt, r_i + v_i * half_dt, v_i + k2_v * half_dt)
        k4_v = f(t_i + dt, r_i + v_i * dt, v_i + k3_v * dt)

        k1_r = v_i
        k2_r = v_i + k1_v * half_dt
        k3_r = v_i + k2_v * half_dt
        k4_r = v_i + k3_v * dt

        v[i + 1] = v_i + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * dt / 6
        r[i + 1] = r_i + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) * dt / 6
        a[i + 1] = f(t[i + 1], r[i + 1], v[i + 1])
        if progressbar:
            progress.progress_bar(i,len(t)-1,time_start)

    return r, v, a

def kinematic_euler(t, f, r0, v0, progressbar = False):
    """
    Simulates the motion of an object using the Forward Euler method
    for systems described by position, velocity, and a time-dependent acceleration function.

    Parameters
    ----------
    t : ndarray
        1D array of time points at which to compute the solution. Assumes uniform spacing.
    f : callable
        Function of the form `f(t, r, v)` returning the acceleration (a vector) at time `t`,
        position `r`, and velocity `v`.
    r0 : ndarray
        Initial position vector.
    v0 : ndarray
        Initial velocity vector.

    Returns
    -------
    r : ndarray
        Array of positions at each time step. Shape: (len(t), len(r0)).
    v : ndarray
        Array of velocities at each time step. Shape: (len(t), len(v0)).
    a : ndarray
        Array of accelerations at each time step. Shape: (len(t), len(v0)).
    """
    if progressbar:
        time_start = time.time()
    dt = t[1] - t[0]
    dim = len(r0)

    r = np.zeros((len(t), dim))
    v = np.zeros((len(t), dim))
    a = np.zeros((len(t), dim))

    r[0] = r0
    v[0] = v0
    a[0] = f(t[0], r0, v0)

    for i in range(len(t) - 1):
        a[i] = f(t[i], r[i], v[i])
        v[i + 1] = v[i] + a[i] * dt
        r[i + 1] = r[i] + v[i] * dt
        a[i + 1] = f(t[i + 1], r[i + 1], v[i + 1])
        if progressbar:
            progress.progress_bar(i,len(t)-1,time_start)
    return r, v, a

def kinematic_rk4_multi(t, f, R0, V0, progressbar=True):
    """
    RK4 integrator for multiple independent bodies under acceleration f(t, R, V).

    Parameters
    ----------
    t : ndarray
        1D time array
    f : callable
        Function f(t, R, V) → A, where R and V are shape (N, D)
    R0 : ndarray
        Initial positions of shape (N, D)
    V0 : ndarray
        Initial velocities of shape (N, D)

    Returns
    -------
    R : ndarray
        Positions at all time steps (T, N, D)
    V : ndarray
        Velocities at all time steps (T, N, D)
    A : ndarray
        Accelerations at all time steps (T, N, D)
    """
    if progressbar:
        time_start = time.time()

    N, D = R0.shape
    dt = t[1] - t[0]
    half_dt = dt / 2
    T = len(t)

    R = np.zeros((T, N, D))
    V = np.zeros((T, N, D))
    A = np.zeros((T, N, D))

    R[0] = R0
    V[0] = V0
    A[0] = f(t[0], R0, V0)

    for i in range(T - 1):
        t_i = t[i]
        R_i = R[i]
        V_i = V[i]

        k1_v = f(t_i, R_i, V_i)
        k2_v = f(t_i + half_dt, R_i + V_i * half_dt, V_i + k1_v * half_dt)
        k3_v = f(t_i + half_dt, R_i + V_i * half_dt, V_i + k2_v * half_dt)
        k4_v = f(t_i + dt, R_i + V_i * dt, V_i + k3_v * dt)

        k1_r = V_i
        k2_r = V_i + k1_v * half_dt
        k3_r = V_i + k2_v * half_dt
        k4_r = V_i + k3_v * dt

        V[i + 1] = V_i + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6
        R[i + 1] = R_i + (k1_r + 2*k2_r + 2*k3_r + k4_r) * dt / 6
        A[i + 1] = f(t[i + 1], R[i + 1], V[i + 1])

        if progressbar:
            progress.progress_bar(i, T - 1, time_start)

    return R, V, A
