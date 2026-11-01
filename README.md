GARCH-from-Scratch: From ARCH(1) to GARCH(1,1)
1. Introduction to Volatility
When dealing with financial volatility, it is easy to see that variance is not constant over time. If we consider historical data and calculate a standard static estimator of the standard deviation, it implies that we face the exact same "risk" every single day—an assumption that is clearly false in real markets.

To model time-varying volatility, we introduce ARCH($p$) models. Let's start by looking at the simplest case: ARCH(1).

2. The ARCH(1) Model
The core idea in ARCH models is that today's volatility depends on the past shocks (perturbations) of our data. In the case of ARCH(1), volatility depends directly on yesterday's shock:

$$\sigma^2_t = \alpha_0 + \alpha_1 \varepsilon^2_{t-1}$$

For the rest of this framework, we consider a filtration ${\mathcal{F}t}$ (the information available up to time $t$) and define log-returns as $y_t = \log(S_t / S{t-1})$, where:

$$y_t = \mu_t + \varepsilon_t$$

$$\varepsilon_t = \sigma_t Z_t, \quad \text{with } Z_t \sim \mathcal{N}(0,1) \text{ i.i.d.}$$

(Remark: To keep things simple, since we are not focusing on time-varying means, we will consider $\mu_t$ as a constant.)

3. Likelihood Function & Optimization
To adjust this model to our data, we need to determine which parameters are most likely by maximizing the likelihood function.

Given that $y_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(\mu_t, \sigma^2_t)$, the conditional probability density function is:

$$f(y_t \mid \mathcal{F}_{t-1}) = \frac{1}{\sqrt{2\pi\sigma^2_t}} \exp\left(-\frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right) = \frac{1}{\sqrt{2\pi\sigma^2_t}} \exp\left(-\frac{\varepsilon_t^2}{2\sigma_t^2}\right)$$

Since $\varepsilon_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(0, \sigma^2_t)$, the distribution depends entirely on $\sigma_t^2$, which in turn relies on our coefficients. The likelihood function is:

$$L(\alpha_0, \alpha_1) = \prod_{t=2}^n f(y_t \mid \mathcal{F}{t-1}) = \prod{t=2}^n \frac{1}{\sigma_t \sqrt{2\pi}} \exp\left(-\frac{\varepsilon_t^2}{2\sigma_t^2}\right)$$

Taking logs, we get the Log-Likelihood function:

$$l(\alpha_0, \alpha_1) = -\frac{1}{2} \sum_{t=2}^n \left( \log(2\pi) + \log(\sigma_t^2) + \frac{\varepsilon_t^2}{\sigma_t^2} \right)$$

Analytical Derivatives
Since $\sigma_t^2 = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2$, the partial derivatives of the variance with respect to the parameters are:

$$\frac{\partial \sigma_t^2}{\partial \alpha_0} = 1, \quad \frac{\partial \sigma_t^2}{\partial \alpha_1} = \varepsilon_{t-1}^2$$

Thus, the partial derivatives of the log-likelihood function are:

$$\frac{\partial l}{\partial \alpha_0}(\alpha_0, \alpha_1) = -\frac{1}{2} \sum_{t=2}^n \frac{1}{\sigma_t^2} \left( 1 - \frac{\varepsilon_t^2}{\sigma_t^2} \right) \frac{\partial \sigma_t^2}{\partial \alpha_0} = -\frac{1}{2} \sum_{t=2}^n \left( \frac{1}{\sigma_t^2} - \frac{\varepsilon_t^2}{\sigma_t^4} \right)$$

$$\frac{\partial l}{\partial \alpha_1}(\alpha_0, \alpha_1) = -\frac{1}{2} \sum_{t=2}^n \frac{1}{\sigma_t^2} \left( 1 - \frac{\varepsilon_t^2}{\sigma_t^2} \right) \frac{\partial \sigma_t^2}{\partial \alpha_1} = -\frac{1}{2} \sum_{t=2}^n \varepsilon_{t-1}^2 \left( \frac{1}{\sigma_t^2} - \frac{\varepsilon_t^2}{\sigma_t^4} \right)$$

Gradient Ascent Algorithm (From Scratch)
Since we are building this from scratch, we use a simple gradient ascent algorithm to find the maximum:

Let $\theta^k = (\alpha_0^k, \alpha_1^k)$. We update our parameters via:

$$\theta^{k+1} = \theta^k + h \nabla l(\theta^k)$$

where $h > 0$ is the step size.
