# te_datasim

Simulate timeseries data that can be used to evaluate transfer entropy (TE) estimation methods.

## Currently implemented models:

### 1.1) Bivariate Linear Gaussian System

`BVLinearGaussianSimulator` implements a simple stochastic coupled system from https://doi.org/10.1063/5.0053519

- Causal graph:

$$
Y \to X
$$

- Equations:

$$
\begin{align*}
			x_{t+1} &= b_x x_t + \lambda y_t + \mathcal{E} \sim \mathcal{N}(0, \sigma^2_x) \\
			y_{t+1} &= b_y y_t + \mathcal{E} \sim \mathcal{N}(0, \sigma^2_y)
\end{align*}
$$

- Optimal solution: Both $T_{X\to Y}$ and $T_{Y \to X}$ have closed form solutions, which are accessible using the `analytic_transfer_entropy` method. A good TE estimator will yield $\hat{T}_{X\to Y}$ and $\hat{T}_{Y\to X}$ close to the analytic values.

-----

### 1.2) Multivariate Linear Gaussian System

`MVLinearGaussianSimulator` implements `n_dim` independent channels of the bivariate linear Gaussian system, allowing for the simulation of vector valued time series data.

- Causal graph:

$$
Y \to X
$$

- Optimal solution: Both $T_{X\to Y}$ and $T_{Y \to X}$ have closed form solutions, which are accessible using the `analytic_transfer_entropy` method. As the channels are independent, this is simply the sum of the TEs for each channel.

-----

### 2) Bivariate Joint Process System

`BVJointProcessSimulator` implements a simple stochastic joint process system from https://doi.org/10.48550/arXiv.1912.07277

- Causal graph:

$$
X \to Y
$$

- Equations:

$$
\begin{align*}
			z_{t+1} &= \mathcal{N}(0, 1) \\
			x_{t+1} &= \mathcal{N}(0, \rho) \\
			y_{t+1} &= 
			\begin{cases}
				z_{t+1} \text{if } y_{t} < \lambda \\
				x_{t} + z_{t}\sqrt{1-\rho^2}
			\end{cases}
\end{align*}
$$

- Optimal solution: Both $T_{X\to Y}$ and $T_{Y \to X}$ have closed form solutions, which are accessible using the `analytic_transfer_entropy` method. A good TE estimator will yield $\hat{T}_{X\to Y}$ and $\hat{T}_{Y\to X}$ close to the analytic values.

-----

### 3) Neural System

`NeuralSimulator` implements a simulation of a neural system as described by the article "Network modelling methods for FMRI" by Smith et al (2011). The causal diagram is:

$$
\text{input signal}
$$

$$
\downarrow
$$

$$
\text{spike counts} \leftarrow \text{neural activity} \to \text {bold signal}
$$



so called binary `input_signals` (across a suitable number of independent channels) are generated using a stochastic switching process with mean durations for the two states being poisson distributed.

`neural_activity` across a specified number of neural regions is then generated in response to these input signals using a discrete time linear differential equation of the form:

$$
y_{t+1} = y_t + \left[\mathbf{A}y_{t-1} + \mathbf{C}x_{t-1} + \mathcal{N}(0, \sigma_r)\right]
$$

Where $\mathbf{A}$ is a neuron-neuron connectivity matrix, and $\mathbf{C}$ defines how each input channel is sent to each neuron. 

A `bold_signal` (corresponding to an fMRI readout for a given neural area) is simulated based on the time series of neural activity using a complex nonlinear differential equation system.

Finally, binned `spike_counts` are also generated using a non-homogenous poisson process with rates scaled by the neural activity. 

- Optimal solution: Closed form solutions for the TEs of this system are generally not known precisely, however, the causal diagram should at least specify which TE values should be positive, and which should be 0.

-----
