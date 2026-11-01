# garch-from-scratch
## 1. Introduction to Volatility
When dealing with financial volatility, it is easy to see that variance is not constant over time. If we calculate a standard static estimator of standard deviation from historical data, it implies a constant level of risk every day—an assumption that is clearly false in real markets.

## 2. The ARCH(1) Model
To capture time-varying volatility, we introduce **Autoregressive Conditional Heteroskedasticity (ARCH)** models. Let's start with the simplest case, the **ARCH(1)** model.

The core idea of ARCH models is that today's volatility depends on past shocks (perturbations) in our data. In the case of an ARCH(1) model, volatility depends directly on yesterday's shock:

$$\sigma^2_t = \alpha_0 + \alpha_1 \varepsilon^2_{t-1}$$

## 3. Statistical Framework & Filtration
Let $\{\mathcal{F}_t\}$ represent the filtration (the information available up to time $t$). We define the log-returns of an asset as $y_t = \log(S_t / S_{t-1})$, decomposed as:

$$y_t = \mu_t + \varepsilon_t$$

Where the error term (or innovation) is scaled by volatility:

$$\varepsilon_t = \sigma_t Z_t, \quad \text{with } Z_t \sim \text{i.i.d. } N(0,1)$$

## 4. Likelihood Function
To fit this model to our data and estimate the parameters, we need to maximize the likelihood function. 

Given the conditional distribution $y_t \mid \mathcal{F}_{t-1} \sim N(\mu_t, \sigma^2_t)$, the conditional probability density function is expressed as:

$$f(y_t \mid \mathcal{F}_{t-1}) = \frac{1}{\sqrt{2\pi\sigma^2_t}} \exp\left(-\frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right) = \frac{1}{\sqrt{2\pi\sigma^2_t}} \exp\left(-\frac{\varepsilon_t^2}{2\sigma_t^2}\right)$$
