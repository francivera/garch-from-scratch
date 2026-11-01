import numpy as np

def exact_gradient(alpha0, alpha1, epsilon):
    n = len(epsilon)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(epsilon)  

    for t in range(1, n):
        sigma2[t] = alpha0 + alpha1 * (epsilon[t - 1] ** 2)

    termino_comun = (1.0 / sigma2) - ((epsilon**2) / (sigma2**2))

    dl_dalpha0 = -0.5 * np.sum(termino_comun[1:])
    dl_dalpha1 = -0.5 * np.sum((epsilon[:-1] ** 2) * termino_comun[1:])

    return np.array([dl_dalpha0, dl_dalpha1])

def ll(alpha0, alpha1, epsilon):
    n = len(epsilon)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(epsilon)

    for t in range(1, n):
        sigma2[t] = alpha0 + alpha1 * epsilon[t-1]**2

    if np.any(sigma2 <= 0):
        return -np.inf

    return -0.5 * np.sum(
        np.log(sigma2[1:]) + epsilon[1:]**2 / sigma2[1:]
    )

def arch(l_func, epsilon, h, tol):
    max_alpha0 = np.var(epsilon)
    iters = int(1e4)
    vbest = -np.inf
    
    for i in range(1, iters):
        alpha0, alpha1 = np.random.uniform(1e-8, max_alpha0), np.random.uniform(1e-8, 0.9999999)
        a = l_func(alpha0, alpha1, epsilon)
        if a > vbest:
            best = np.array([alpha0, alpha1])
            vbest = a
            
    theta = best
    n = 1
    er = tol + 1
    
    while er > tol and n < iters:
        theta = best
        alpha0, alpha1 = theta[0], theta[1]
        delta = exact_gradient(alpha0, alpha1, epsilon)
        best = theta + h * delta
        er = np.linalg.norm(best - theta)
        
        if best[0] <= 0 or best[1] < 0 or best[1] >= 1 or l_func(best[0], best[1], epsilon) <= vbest:
            best = theta
            n += 1
            h /= 2
        else:
            if abs(vbest - l_func(best[0], best[1], epsilon)) < 1e-14:
                break
            vbest = l_func(best[0], best[1], epsilon)
            h *= 1.5
            n += 1
            
    return best