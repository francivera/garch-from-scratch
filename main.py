import numpy as np
import yfinance as yf
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from arch_model_scratch import l, arch

# 1. Descargar datos (ej. Apple / AAPL)
datos = yf.download("AAPL", period="1y", auto_adjust=True)

# 2. Precios de cierre y rendimientos logarítmicos
precios = datos["Close"].dropna()
if isinstance(precios, pd.DataFrame):
    precios = precios.iloc[:, 0]
returns = np.log(precios / precios.shift(1)).dropna()

epsilon = (returns - returns.mean()).to_numpy().ravel()

# 3. Ejecutar algoritmo ARCH desde cero
resultado = arch(l, epsilon, h=1e-4, tol=1e-8)
alpha_0, alpha_1 = resultado[0], resultado[1]

print(f"Alpha 0 (Scratch): {alpha_0:.6f}")
print(f"Alpha 1 (Scratch): {alpha_1:.6f}")
print("Log-verosimilitud:", l(alpha_0, alpha_1, epsilon))

# 4. Calcular volatilidad estimada para ARCH(1)
n = len(epsilon)
sigma2 = np.zeros(n)
sigma2[0] = np.var(epsilon)

for t in range(1, n):
    sigma2[t] = alpha_0 + alpha_1 * (epsilon[t - 1] ** 2)

volatilidad = np.sqrt(sigma2)

# 5. Gráfica comparativa de rendimientos y volatilidad
plt.figure(figsize=(12, 6))
plt.plot(epsilon, color="dodgerblue", alpha=0.6, label="Rendimientos ($\epsilon_t$)", linewidth=0.8)
plt.plot(volatilidad, color="crimson", linewidth=1.5, label="Volatilidad estimada ($+\sigma_t$)")
plt.plot(-volatilidad, color="crimson", linewidth=1.5, label="$-\sigma_t$")
plt.title(f"Modelo ARCH(1) From Scratch — AAPL ($\\alpha_0={alpha_0:.4f}, \\alpha_1={alpha_1:.4f}$)", fontsize=14)
plt.xlabel("Días bursátiles", fontsize=11)
plt.ylabel("Porcentaje (%)", fontsize=11)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# 6. Comparación con la librería oficial 'arch' (GARCH(1,1))
modelo = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
resultado_garch = modelo.fit(disp="off")
print(resultado_garch.summary())

volatilidad_garch = resultado_garch.conditional_volatility

plt.figure(figsize=(12, 5))
plt.plot(volatilidad_garch, color="darkorange", linewidth=1.5, label="Volatilidad GARCH(1,1) ($\sigma_t$) [Librería arch]")
plt.title("Evolución de la Volatilidad Estimada — GARCH(1,1) Oficial (AAPL)", fontsize=14)
plt.xlabel("Días bursátiles", fontsize=11)
plt.ylabel("Volatilidad (%)", fontsize=11)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()