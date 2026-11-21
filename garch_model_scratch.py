import numpy as np
def exact_gradient(alpha0, alpha1, beta, epsilon):
    n = len(epsilon)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(epsilon)

    dsigma2_dalpha0 = np.zeros(n)
    dsigma2_dalpha1 = np.zeros(n)
    dsigma2_dbeta = np.zeros(n)

    for t in range(1, n):
        sigma2[t] = alpha0 + alpha1 * (epsilon[t - 1] ** 2) + beta * sigma2[t - 1]

        dsigma2_dalpha0[t] = 1.0 + beta * dsigma2_dalpha0[t - 1]
        dsigma2_dalpha1[t] = (epsilon[t - 1] ** 2) + beta * dsigma2_dalpha1[t - 1]
        dsigma2_dbeta[t] = sigma2[t - 1] + beta * dsigma2_dbeta[t - 1]

    termino_comun = (1.0 / sigma2) - ((epsilon**2) / (sigma2**2))

    dl_dalpha0 = -0.5 * np.sum(termino_comun[1:] * dsigma2_dalpha0[1:])
    dl_dalpha1 = -0.5 * np.sum(termino_comun[1:] * dsigma2_dalpha1[1:])
    dl_dbeta = -0.5 * np.sum(termino_comun[1:] * dsigma2_dbeta[1:])

    return np.array([dl_dalpha0, dl_dalpha1, dl_dbeta])
    
def llg(alpha0,alpha1,beta,epsilon):
    n=len(epsilon)
    sigma2=np.zeros(n)
    sigma2[0]=np.var(epsilon)
    for t in range(1,n):
        sigma2[t]=alpha0+alpha1*(epsilon[t-1]**2)+beta*sigma2[t-1]
    if np.any(sigma2 <= 0):
        return -np.inf
    return -0.5 * np.sum(
        np.log(2 * np.pi)+np.log(sigma2[1:]) + epsilon[1:]**2 / sigma2[1:]
        )

def garch(l_func, epsilon, h, tol):
    max_alpha0 = np.var(epsilon)
    iters = int(1e4)
    vbest = -np.inf

    for i in range(1, iters):
        beta = np.random.uniform(1e-8, 0.9999999)
        alpha0 = np.random.uniform(1e-8, max_alpha0)
        alpha1 = np.random.uniform(1e-8, 1 - beta - 1e-6)   # asegura alpha1 + beta < 1
        a = l_func(alpha0, alpha1, beta, epsilon)
        if np.isfinite(a) and a > vbest:
            best = np.array([alpha0, alpha1, beta])
            vbest = a

    theta = best
    n = 1
    er = tol + 1
    

    while er > tol and n < iters:
        theta = best
        alpha0, alpha1, beta = theta[0], theta[1], theta[2]
        delta = exact_gradient(alpha0, alpha1, beta, epsilon)
        best = theta + h * delta
        er = np.linalg.norm(best - theta)

        nueva = l_func(best[0], best[1], best[2], epsilon)

        if (best[0] <= 0 or best[1] < 0 or best[2] < 0
                or best[1] + best[2] >= 1
                or not np.isfinite(nueva)
                or nueva <= vbest):
            best = theta
            n += 1
            h /= 2
        else:
            if abs(vbest - nueva) < 1e-14:
                break
            vbest = nueva
            h *= 1.5
            n += 1

    return best