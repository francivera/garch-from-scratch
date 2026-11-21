import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
from arch_model_scratch import ll, arch
from garch_model_scratch  import garch,llg
import pandas as pd

datos = yf.download("AAPL", period="6y", auto_adjust=True)

precios = datos["Close"].dropna()

returns = np.log(precios / precios.shift(1)).dropna()

epsilon = (returns - returns.mean()).to_numpy().ravel()

resultado = arch(ll, epsilon, h=1e-4, tol=1e-8)
alpha_0, alpha_1 = resultado[0], resultado[1]

print(f"Alpha 0 (Scratch): {alpha_0:.6f}")
print(f"Alpha 1 (Scratch): {alpha_1:.6f}")
print("Log-verosimilitud:", ll(alpha_0, alpha_1, epsilon))

# Reference model
modelo_ref = arch_model(returns, vol="ARCH", p=1, mean="Constant", dist="Normal")
resultado_ref = modelo_ref.fit(disp="off")
omega_ref = resultado_ref.params['omega']
alpha1_ref = resultado_ref.params['alpha[1]']

n = len(epsilon)
sigma2 = np.zeros(n)
sigma2[0] = np.var(epsilon)

for t in range(1, n):
    sigma2[t] = alpha_0 + alpha_1 * (epsilon[t - 1] ** 2)

volatilidad_scratch = np.sqrt(sigma2)

# Introducing the real dates
volatilidad_ref = resultado_ref.conditional_volatility

plt.figure(figsize=(12, 6))

plt.plot(
    returns.index, 
    volatilidad_scratch, 
    color="crimson", 
    linewidth=1.5, 
    label=f"From Scratch: $\\alpha_0={alpha_0:.4f}, \\alpha_1={alpha_1:.4f}$"
)

plt.plot(
    volatilidad_ref.index,  
    volatilidad_ref,      
    color="darkorange", 
    linewidth=1.5, 
    linestyle="--", 
    label=f"Reference (arch): $\\omega={omega_ref:.4f}, \\alpha_1={alpha1_ref:.4f}$"
)

plt.title("Volatility comparison in ARCH(1) From Scratch vs. Official library", fontsize=14)
plt.xlabel("Fecha", fontsize=11)
plt.ylabel("Estimated volatility ($\sigma_t$)", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()

#%%
scale = 100.0
eps_s = epsilon * scale
eps_series = pd.Series(eps_s, index=returns.index)

modelo_ref = arch_model(
    eps_series, vol="GARCH", p=1, q=1, mean="Zero", dist="Normal", rescale=False
)
resultado_ref = modelo_ref.fit(disp="off")

# Desescalar parámetros de varianza (omega está en escala^2) y volatilidad (en escala^1)
omega_ref_s = resultado_ref.params["omega"]
omega_ref = omega_ref_s / (scale ** 2)
alpha1_ref = resultado_ref.params["alpha[1]"]
beta1_ref = resultado_ref.params["beta[1]"]

# Estimación from scratch
theta = garch(llg, epsilon, h=1e-4, tol=1e-8)
alpha0_hat, alpha1_hat, beta_hat = theta[0], theta[1], theta[2]

n = len(epsilon)
sigma2 = np.zeros(n)
sigma2[0] = np.var(epsilon)

for t in range(1, n):
    sigma2[t] = (
        alpha0_hat
        + alpha1_hat * (epsilon[t - 1] ** 2)
        + beta_hat * sigma2[t - 1]
    )

volatilidad_scratch = np.sqrt(sigma2)
volatilidad_ref = resultado_ref.conditional_volatility / scale

plt.figure(figsize=(12, 6))

plt.plot(
    returns.index,
    volatilidad_scratch,
    color="crimson",
    linewidth=1.5,
    label=(
        f"From Scratch: $\\alpha_0={alpha0_hat:.2e}, \\alpha_1={alpha1_hat:.4f}, "
        f"\\beta={beta_hat:.4f}$"
    ),
)

plt.plot(
    volatilidad_ref.index,
    volatilidad_ref,
    color="darkorange",
    linewidth=1.5,
    linestyle="--",
    label=(
        f"Reference (arch): $\\omega={omega_ref:.2e}, \\alpha_1={alpha1_ref:.4f}, "
        f"\\beta={beta1_ref:.4f}$"
    ),
)

plt.title(
    "Volatility comparison in GARCH(1,1) From Scratch vs. Official library",
    fontsize=14,
)
plt.xlabel("Fecha", fontsize=11)
plt.ylabel("Estimated volatility ($\\sigma_t$)", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
