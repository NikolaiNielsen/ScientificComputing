import numpy as np
from progress.bar import Bar


def gss(f, a, b, evaluator=None, max_iter=500, epsilon=1e-6):
    """
    Golden section search for finding the minimum of a 1D unimodal function.
    """

    # Optional evaluator function, if f for example expects vector
    if evaluator is not None:
        ev = evaluator[0]
        args = evaluator[1:]
    else:
        def ev(x, args):
            return x
        args = None

    a, b = min(a, b), max(a, b)
    tau = (np.sqrt(5)-1)/2
    x1 = a + (1-tau)*(b-a)
    f1 = f(ev(x1, args))
    x2 = a+tau*(b-a)
    f2 = f(ev(x2, args))
    for i in range(max_iter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a+tau*(b-a)
            f2 = f(ev(x2, args))
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a+(1-tau)*(b-a)
            f1 = f(ev(x1, args))

        if abs((b-a)) <= epsilon:
            break
    return (b+a)/2


def newton_raphson(f, x0, evaluator=None, h=5e-2, max_iter=50, epsilon=1e-3):
    """
    Newton Raphson method for 1D minimizing.
    """
    if evaluator is not None:
        ev = evaluator[0]
        args = evaluator[1:]
    else:
        def ev(x, args):
            return x
        args = None

    x = [x0]

    for i in range(max_iter):
        fx = f(ev(x[-1], args))
        fminus = f(ev(x[-1]-h, args))
        fplus = f(ev(x[-1]+h, args))
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


def conjugate_gradient(f, x0, g=None, h=1e-4, alpha_max=1,
                       max_iter=100, epsilon=1e-6):
    """
    Conjugate Gradient method for unconstrained optimization
    Uses a numerical approximation to Newtons method for optimization.

    inputs:
    - f: objective function
    - g: gradient of f
    - x0: initial guess
    """
    def evaluator(alpha, args):
        x0, s = args
        return x0 + alpha*s

    if g is None:
        g = num_gradient
    g_last = g(f, x0, h)

    alpha_min = 0
    alpha_max = alpha_max

    s = -g_last
    x_last = x0
    x = [x0]

    bar = Bar("Simulating", max=max_iter)
    for k in range(max_iter):
        alpha = gss(f, alpha_min, alpha_max, [evaluator, x_last, s])
        x_new = x_last + alpha * s
        x.append(x_new)

        bar.next()
        res = x_new-x_last
        # print(res)
        if np.sqrt(np.sum(res**2)) < epsilon:
            break
        x_last = x_new
        g_new = g(f, x_new, h)
        g_new_flat = g_new.flatten()
        g_last_flat = g_last.flatten()
        beta = g_new_flat.dot(g_new_flat) / g_last_flat.dot(g_last_flat)
        s = -g_new + beta * s
    bar.finish()
    return x[:k+1]


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
