# te_datasim

Simulate timeseries data that can be used to evaluate transfer entropy (TE) estimation methods.

## Currently implemented models:

### 1) Bivariate Linear Gaussian System

`BVLinearGaussianSimulator` implements a simple stochastic coupled system with the following causal graph:

$$
Y \to X
$$

The equations that define the system are:

$$
\begin{align*}
			x_{t+1} &= b_x x_t + \lambda y_t + \mathcal{E} \sim \mathcal{N}(0, \sigma^2_x) \\
			y_{t+1} &= b_y y_t + \mathcal{E} \sim \mathcal{N}(0, \sigma^2_y)
\end{align*}
$$

- Optimal solution: Both $T_{X\to Y}$ and $T_{Y \to X}$ have closed form solutions, which are accessible using the `analytic_transfer_entropy` method. A good TE estimator will yield $\hat{T}_{X\to Y}$ and $\hat{T}_{Y\to X}$ close to the analytic values.

-----

### 2) Neural System

`NeuralSimulator` implements a simulation of a neural system as described by the article "Network modelling methods for FMRI" by Smith et al (2011). The causal diagram is:

$$
\text{input signal} \\ \downarrow \\  \text{spike counts} \leftarrow \text{neural activity} \to \text {bold signal}
$$



so called binary `input_signals` (across a suitable number of independent channels) are generated using a stochastic switching process with mean durations for the two states being poisson distributed.

`neural_activity` across a specified number of neural regions is then generated in response to these input signals using a discrete time linear differential equation of the form:

$$
y_{t+1} = y_t + \left[\bold{A}y_{t-1} + \bold{C}x_{t-1} + \mathcal{N}(0, \sigma_r)\right]
$$

Where $\mathbf{A}$ is a neuron-neuron connectivity matrix, and $\mathbf{C}$ defines how each input channel is sent to each neuron. 

A `bold_signal` (corresponding to an fMRI readout for a given neural area) is simulated based on the time series of neural activity using a complex nonlinear differential equation system.

Finally, binned `spike_counts` are also generated using a non-homogenous poisson process with rates scaled by the neural activity. 

- Optimal solution: Closed form solutions for the TEs of this system are generally not known precisely, however, the causal diagram should at least specify which TE values should be positive, and which should be 0.

-----
