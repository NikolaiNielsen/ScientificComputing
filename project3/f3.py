import numpy as np


def newton_raphson(f, x0, h=5e-2, max_iter=50, epsilon=1e-3):
    x = [x0]
    for i in range(max_iter):
        fx = f(x[-1])
        fminus = f(x[-1]-h)
        fplus = f(x[-1]+h)
        diff = (fplus-fminus)/(2*h)
        x_new = x[-1] - fx/diff
        x.append(x_new)
        res = abs((x[-1] - x[-2])/x[-2])
        if res <= epsilon:
            break

    return np.array(x)


def bisection(f, a, b, epsilon=1e-6):
    a, b = min(a, b), max(a, b)
    while b-a > epsilon:
        m = a + (b-a)/2
        fa = f(a)
        fm = f(m)
        sfa = fa/abs(fa)
        fsm = fm/abs(fm)
        if sfa == fsm:
            a = m
        else:
            b = m
    return m


def secant(f, x0, x1, max_iter=100, epsilon=1e-6):
    x = [x0, x1]
    fx = [f(x0), f(x1)]

    for i in range(max_iter):
        x_new = x[-1] - fx[-1]*(x[-1] - x[-2])/(fx[-1] - fx[-2])
        fx.append(f(x_new))
        x.append(x_new)
        if abs((x[-1] - x[-2])/x[-2]) < epsilon:
            break
    return x


def inverse_quadratic(f, a, b, c, max_iter=100, epsilon=1e-6):
    # Sort a, b and c
    # a, b, c = sorted([a, b, c])
    guesses = []
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)
        fc = f(c)

        u = fb/fc
        v = fb/fa
        w = fa/fc
        p = v * (w * (u-w) * (c-b) - (1-u) * (b-a))
        q = (w-1) * (u-1) * (v-1)

        if abs(p/q) < epsilon:
            break

        a, b, c = b, b + p/q, a
        guesses.append(b)

    return guesses


def conjugate_gradient(f, x0, g=None, alpha_0=0.5, h=1e-4,
                       max_iter=100, epsilon=1e-6):
    """
    Conjugate Gradient method for unconstrained optimization
    Uses a numerical approximation to Newtons method for optimization.

    inputs:
    - f: objective function
    - g: gradient of f
    - x0: initial guess
    """
    if g is None:
        g = num_gradient
    g_last = g(f, x0, h)

    s = -g_last
    x_last = x0
    x = [x0]
    for k in range(max_iter):
        alpha = newton_gradient(f, alpha_0, x_last, s, h)
        x_new = x_last + alpha * s
        x.append(x_new)
        res = x_new-x_last
        # print(res)
        if np.sqrt(np.sum(res**2)) < epsilon:
            break
        x_last = x_new
        g_new = g(f, x_new, h)
        beta = g_new.dot(g_new) / g_last.dot(g_last)
        s = -g_new + beta * s
    return x


def newton_gradient(f, alpha, x0, s, h=1e-3, max_iter=100, epsilon=1e-3):
    x_last = alpha
    x_new = alpha
    for i in range(max_iter):
        fx = f(x0 + x_last*s)
        fminus = f(x0 + (x_last-h)*s)
        fplus = f(x0 + (x_last+h)*s)
        x_new = x_last - 0.5*h*(fplus - fminus) / (fplus + fminus - 2*fx)
        res = abs(x_new-x_last)
        if res <= epsilon:
            break
        x_last = x_new

    return x_new


def num_gradient(f, x, h=1e-4):
    """
    computes a numerical gradient of f(x) at the point x
    """
    grad = np.zeros(x.shape)
    h_vec = h*np.eye(x.size)
    for i in range(x.size):
        xp = x+h_vec[i]
        xm = x-h_vec[i]
        fp = f(xp)
        fm = f(xm)
        grad[i] = (fp - fm)/(2*h)
    return grad
