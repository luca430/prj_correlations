import numpy as np

def runge_kutta(f, x0, T, dt = 0.01, **kwargs):
    """
    Solves the differential equation dx/dt = f(x,t) using the 4th-order Runge-Kutta method.

    Parameters:
    - f: Function that computes the derivative (dx/dt) given the current state x at time t.
    - x0: Initial state vector.
    - T: Final time.
    - dt: Time step size.
    - **kwargs: additional arguments of the force f.

    Returns:
    - t_values: Array of time points.
    - x_values: Array of state vectors corresponding to the time points.
    """
    # Initialize time and state
    t_values = np.arange(0, T, dt)
    x_values = np.zeros((len(t_values), len(x0)))

    x = np.array(x0)
    x_values[0] = x

    # Runge-Kutta iteration
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        
        k1 = dt * f(x, t, **kwargs)
        k2 = dt * f(x + 0.5 * k1, t, **kwargs)
        k3 = dt * f(x + 0.5 * k2, t, **kwargs)
        k4 = dt * f(x + k3, t, **kwargs)
        
        x = x + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_values[i] = x

    return t_values, x_values
