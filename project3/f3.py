import numpy as np
from scipy.optimize import line_search


def gss(f, a, b, evaluator=None, max_iter=500, epsilon=1e-6):
    """
    Golden section search for finding the minimum of a 1D unimodal function.
    """

    # Optional evaluator function, if f for example expects vector
    # (line search)
    if evaluator is not None:
        ev = evaluator[0]
        args = evaluator[1:]
    else:
        def ev(x, args):
            return x
        args = None

    # initialize parameters
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
    Newton Raphson method for 1D root finding. Uses a numerical derivative
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
    """
    Bisection search for locating a root. Requires sign(f(a)) =/= sign(f(b))
    """
    a, b = min(a, b), max(a, b)

    # Test for whether convergence will happen
    fa = f(a)
    fb = f(b)
    if fa/abs(fa) == fb/abs(fb):
        return None

    # Main algorithm
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
    """
    Secant algorithm for root finding.
    """
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
    """
    Inverse quadratic interpolation for root finding.
    """
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


def conjugate_gradient(f, x0, g=None, g2=None, alpha_max=1, n_restart=10,
                       max_iter=100, epsilon=1e-6, return_s=False):
    """
    Conjugate Gradient method for unconstrained optimization
    Uses a numerical gradient if none is provided. Uses GSS for line search.

    inputs:
    - f: objective function
    - g: gradient of f
    - x0: initial guess
    """
    def evaluator(alpha, args):
        x0, s = args
        return x0 + alpha*s
    if g is None:
        h = 1e-4
        g = num_gradient
    g_last, max_last = g(x0)

    alpha_min = 0

    s = -g_last
    x = [x0]
    ss = [s]
    alphas = []

    # bar = Bar("Simulating", max=max_iter)
    for k in range(max_iter):
        alpha = gss(f, alpha_min, alpha_max, [evaluator, x[-1], ss[-1]])
        alphas.append(alpha)
        x_new = x[-1] + alpha * ss[-1]
        x.append(x_new)
        # bar.next()
        res = x[-1] - x[-2]
        if np.sum(np.sqrt(np.sum(res**2))) < epsilon:
            break

        G, max_new = g(x[-1])
        if k+1 % n_restart:
            beta = 0
        else:
            # We include this factor to account for the gradient normalization
            factor = max_new/max_last
            beta = factor*factor*G.flatten().dot(G.flatten()) / \
                (ss[-1].flatten().dot(ss[-1].flatten()))

        s = -G + beta*ss[-1]
        ss.append(s)
    # bar.finish()
    if return_s:
        return x[:k+1], ss[:k+1]
    return x[:k+1]


def newton_gradient(f, alpha, x0, s, h=1e-3, max_iter=100, epsilon=1e-3):
    """
    Newtons method for 1D minimizing. Uses numerical derivatives.
    """
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


def neighbour(x, stepsize=1):
    # Pick a random parameter, and add a random amount (uniform) centred around
    # 0, with size stepsize.
    choice = np.random.randint(x.size)
    x[choice] += np.random.uniform(-stepsize/2, stepsize/2)
    return x


def anneal(f, x0, neighbour=neighbour, neigharg=1, NT=100, Nf=100, target=1e-3,
           minimize=True):

    if not minimize:
        # If we wanna maximize, we just change the sign of the cost function
        # and target.
        def neigh(x, arg):
            return -neighbour(x, arg)
        target = -target
    else:
        neigh = neighbour

    # Set up starting parameters
    x = x0
    cost = f(x)
    xs = [x]
    costs = [cost]

    # Initial temperature and final temperature
    T = 1.0
    T_min = 1e-5
    # Scaling factor for T, chosen to temperature is changed NT times
    alpha = (T_min/T)**(1/NT)

    while T > T_min:
        # For each temperature we test Nf neighbours.
        for _ in range(Nf):
            x_new = neigh(x, neigharg)
            cost_new = f(x_new)

            # if cost_new is better than cost, it is automatically accepted,
            # since ap >= 1
            ap = np.exp((cost-cost_new)/T)
            if ap > np.random.uniform():
                x = x_new
                cost = cost_new
                costs.append(cost)
                xs.append(x)

        # Decrease temperature
        T = T * alpha

        if cost <= target:
            break
    return xs, costs
