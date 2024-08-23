import numpy as np


def differentiate(u, dt):
    N_t = len(u)
    discrete_derivative = np.zeros(len(u))

    # Handle edge cases with forward and backward differences
    discrete_derivative[0] = (u[1] - u[0]) / dt

    discrete_derivative[-1] = (u[-1] - u[-2]) / dt

    # Centered difference
    for n in range(1, N_t-1):
        discrete_derivative[n] = (u[n+1] - u[n-1]) / (2 * dt)
    return discrete_derivative

def differentiate_vector(u, dt):
    d = np.zeros(len(u))

    # Handle edge cases with forward and backward differences
    d[0] = (u[1] - u[0]) / dt

    d[-1] = (u[-1] - u[-2]) / dt
    
    # Centered difference
    d[1:-1] = (u[2:] -u[0:-2])/(2*dt)
    return d

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    