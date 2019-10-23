import numpy as np


def linspace_with_ghosts(a, b, n):
    """
    Returns a vector x with n linearly spaced points between a and b, along
    with a ghost node at each end, with the same linear spacing
    """
    # the spacing
    dx = (b-a)/(n-1)
    # then we want n+2 points between a-dx and b+dx:
    x = a + dx * np.arange(0, n+2) - dx
    return x, dx


def plot_without_ghosts(x, y, z, ax, **kwargs):
    """
    Plots a 2D surface on an axis object ax, without plotting the ghost nodes
    """
    ax.plot_surface(x[1:-1, 1:-1], y[1:-1, 1:-1], z[1:-1, 1:-1], **kwargs)
    return None


def f(x, params):
    """
    Differential equation for HIV transmittance.
    simulates for n different set of parameters

    Inputs:
    x - [n, 4]
    params - [n, 17]
    """
    a1, a2, b1, b2, b3, c1, c2, d1, e, r1, r2, r3, r4, p1, p2, q, r = params.T
    x1, x2, y, z = x.T
    dx1 = a1*x1*(p1-x1) + a2*x2*(p1-x1) - r1*x1
    dx2 = b1*x1*(p2-x2) + b2*x2*(p2-x2) + b3*y*(p2-x2) - r2*x2
    dy = c1*x2*(q-y) + c2*z*(q-y) - r3*y
    dz = d1*y*(r-z) + e*x1*(r-z) - r4*z

    diffs = np.zeros(x.shape)
    diffs[:, 0] = dx1
    diffs[:, 1] = dx2
    diffs[:, 2] = dy
    diffs[:, 3] = dz
    return diffs


def multivariate_newton(f, J, x0, params=None, tol=1e-6, maxiter=100,
                        return_k=False):
    x0 = np.atleast_2d(x0)
    params = np.atleast_2d(params)
    xk = x0
    for k in range(maxiter):
        Jk = J(x0, params)
        fk = -f(x0, params).squeeze()
        sk = np.linalg.solve(Jk, fk)
        xk = x0 + sk
        res = xk-x0
        if np.sqrt(np.sum(res**2)) <= tol:
            break
        x0 = xk
    if return_k:
        return xk, k+1
    return xk


def rk4(r, params, dt, f):
    # The four contributions for the next step in the simulation
    k1 = f(r, params)
    k2 = f(r + 0.5*dt*k1, params)
    k3 = f(r + 0.5*dt*k2, params)
    k4 = f(r + dt*k3, params)
    return r + dt * (k1 + 2*k2 + 2*k3 + k4)/6


def forward_euler(r, params, dt, f):
    """
    Forwards Euler method.
    """
    dx = f(r, params)
    return r + dt * dx


def sim(f, x0, params, dt=5e-4, N=350, sim_method=rk4):
    """
    Simulate the system of equations using RK4, for n sets of parameters.

    f must have signature, dx = f(x, params),
    sim_method must have signature, x_next = sim_method(x_last, params, dt, f)
    """
    # create our position vector. Assume x0.shape == (n, 4), where n is number
    # of sets of parameters. Returns a [n, 4, N] array. First axis is parameter
    # set/initial condition, second axis is coordinate number, third axis is
    # time.
    x = np.zeros(list(np.atleast_2d(x0).shape) + [N])
    x[:, :, 0] = x0
    t = np.cumsum(np.zeros(N)+dt) - dt
    for n in range(1, N):
        x[:, :, n] = sim_method(x[:, :, n-1], params, dt, f)
    return x, t
