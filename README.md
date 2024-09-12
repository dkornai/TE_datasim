# te_datasim

Simulate timeseries data that can be used to evaluate transfer entropy (TE) estimation methods. 

## General Workflow:

The `Simulator` class represents the data generating process. The data generating process has two or more random variables [which each may be 1D or multi-D]. The causal relationships between the variables have fixed direction, but controllable strength. The strength of this interaction is measured by transfer entropy.

The `Simulator`  class has three methods, corresponding to the three major steps in the workflow.

1) initialise an instance of a given simulator sub-class. This specifies the numeric parameters for the generative process. 

2) generate data using the `.simulate(time=, seed=)` method, which returns a tuple of numpy arrays, each corresponding to an observed time series for a given system variable. 

3) (optional) get a reference analytically computed TE, using the `.get_analytic_transfer_entropy(source=, dest=)` method

Use your transfer entropy estimator of choice on the generated dataset, and compare the results to the analytic reference.

## Simulator Subclasses:

### 1) Bivariate Linear Gaussian System

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

- Strenth of causal interaction (TE) is controlled mainly by the $\lambda$ parameter. 

-----

### 1A) Multivariate Linear Gaussian System

`MVLinearGaussianSimulator` implements `n_dim` independent channels of the bivariate linear Gaussian system, allowing for the simulation of vector valued time series data. Transfer entropy scales linearly with `n_dim`.

The data can also be concatenated with a further `n_redundant_dim` of i.i.d. noise, which, should leave the true TE unchanged. 

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

- Strength if causal interaction (TE) is controlled mainly by the $\lambda$ parameter. 

-----

### 2A) Multivariate Joint Process System

`MVJointProcessSimulator` implements `n_dim` independent channels of the bivariate linear Gaussian system, allowing for the simulation of vector valued time series data. Transfer entropy scales linearly with `n_dim`.

The data can also be concatenated with a further `n_redundant_dim` of i.i.d. noise, which, should leave the true TE unchanged.

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

so called binary `input_signals` (across a suitable number of independent channels) are generated using a stochastic switching process with mean durations for the two states being exponentially distributed.

`neural_activity` across a specified number of neural regions is then generated in response to these input signals using a discrete time linear differential equation of the form:

$$
y_{t+1} = y_t + \left[\mathbf{A}y_{t-1} + \mathbf{C}x_{t-1} + \mathcal{N}(0, \sigma_r)\right]
$$

Where $\mathbf{A}$ is a neuron-neuron connectivity matrix, and $\mathbf{C}$ defines how each input channel is sent to each neuron. 

A `bold_signal` (corresponding to an fMRI readout for a given neural area) is simulated based on the time series of neural activity using a complex nonlinear differential equation system.

Finally, binned `spike_counts` are also generated using a non-homogenous poisson process with rates scaled by the neural activity. 

Closed form solutions for the TEs of this system are generally not known precisely, however, the causal diagram should at least specify which TE values should be positive, and which should be 0.

-----
