import numpy as np

class BVLinearGaussianSimulator():
    """
    BiVariate Linear Gaussian Simulator
    DAG: Y -> X

    System is described by the following set of equations:

    X[t+1] = b_x * X[t] + coupling * Y[t] + normal(0, sqrt(var_x)) \\
    Y[t+1] = b_y * Y[t] + normal(0, sqrt(var_y))

    """
    def __init__(self, coupling = 0.5, b_x = 0.8, b_y = 0.4, var_x = 0.2, var_y = 0.2):
        assert isinstance(coupling, float)
        assert isinstance(b_x, float)
        assert isinstance(b_y, float)
        assert isinstance(var_x, float)
        assert isinstance(var_y, float)
        assert coupling >= 0
        assert var_x >= 0
        assert var_y >= 0

        # parameters of the linear system
        self.coupling = coupling
        self.b_x = b_x
        self.b_y = b_y
        self.var_x = var_x
        self.var_y = var_y

        # attributes
        self.variables = ['X', 'Y']

    def simulate(
            self, 
            time, 
            seed:       int
            ) ->        tuple[np.ndarray, np.ndarray]:
        """
        Simulates a stochastic coupled linear system.
        
        Returns:
        -------
        X : np.ndarray
            time series of variable X 
        Y : np.ndarray
            time series of variable Y
        """
        assert isinstance(time, int)
        assert time > 0
        assert isinstance(seed, int)

        N_discard = 100 # discard first 100 values

        np.random.seed(seed)
        
        # Random initialisation
        x = np.zeros((N_discard + time, ))
        y = np.zeros((N_discard + time, ))
        x[0] = np.random.normal(scale = 0.1)
        y[0] = np.random.normal(scale = 0.1)
        
        # Simulate
        for ii in range(N_discard + time - 1):
            x[ii + 1] = self.b_x * x[ii] + self.coupling * y[ii] + np.random.normal(scale = np.sqrt(self.var_x))
            y[ii + 1] = self.b_y * y[ii] + np.random.normal(scale = np.sqrt(self.var_y))
        
        # Discard and format
        X = x[N_discard:].reshape(-1, 1)
        Y = y[N_discard:].reshape(-1, 1)

        return  X, Y
    
    def analytic_transfer_entropy(
            self, 
            var_from:   str, 
            var_to:     str
            ) ->        float:
        """
        Analytic Transfer Entropy Calculation for the bivariate linear Gaussian model

        TE Y->X is given in Equation 17 in the supplementary materials of [10.1063/5.0053519] 
        TE X->Y is always 0
        
        Parameters:
        ----------
        var_from : str
            The variable transferring information to the variable of interest
        var_to : str
            The variable of interest

        Returns:
        -------
        float
            transfer entropy of the system, given the parameters
        """
        
        assert var_from in self.variables, f"var_from must be in {self.variables}"
        assert var_to in self.variables, f"var_to must be in {self.variables}"
        assert var_from != var_to, "var_from and var_to must be different variables"
    
        if var_from == 'Y' and var_to == 'X':
            var_x = self.var_x
            var_y = self.var_y
            var_x2 = var_x**2
            var_y2 = var_y**2

            b_x = self.b_x
            b_y = self.b_y
            b_x2 = b_x**2
            b_y2 = b_y**2

            lam = self.coupling

            num = ((1-b_y2)*((1-(b_x*b_y))**2)*(var_x**4)) + ((2*(lam**2))*(1-(b_x*b_y))*(var_x2)*(var_y2)) + ((lam**4)*(var_y**4))
            den = ((1-b_y2)*((1-(b_x*b_y))**2)*(var_x**4)) + ((lam**2)*((1-(b_x2*b_y2))*(var_x2)*(var_y2)))
            
            TE_Y2X = 0.5 * np.log(num/den)

            return np.round(TE_Y2X, 4)
        
        elif var_from == 'X' and var_to == 'Y':
            return 0.0
        
import numpy as np

class MVLinearGaussianSimulator():
    """
    MultiVariate Linear Gaussian Simulator (X and Y are vector valued)
    DAG: Y -> X 

    System is simply n_dim independent instances of the BVLinearGaussianSimulator
    """
    def __init__(self, coupling = 0.5, b_x = 0.8, b_y = 0.4, var_x = 0.2, var_y = 0.2, n_dim = None):
        if all(isinstance(i, float) for i in [coupling, b_x, b_y, var_x, var_y]):
            assert n_dim is not None, 'n_dim must be specified as the number of independent duplicate dimensions if all parameters are floats'
            assert isinstance(n_dim, int)
            assert n_dim > 0
            coupling = [coupling] * n_dim
            b_x = [b_x] * n_dim
            b_y = [b_y] * n_dim
            var_x = [var_x] * n_dim
            var_y = [var_y] * n_dim

        if all(isinstance(i, list) for i in [coupling, b_x, b_y, var_x, var_y]):
            assert len(coupling) == len(b_x) == len(b_y) == len(var_x) == len(var_y), 'all parameters must have the same length'
            n_dim = len(coupling)
        
        self.n_dim = n_dim

        # parameters of the linear system, list holds parameters for each independent process
        self.coupling_arr = coupling
        self.b_x_arr = b_x
        self.b_y_arr = b_y
        self.var_x_arr = var_x
        self.var_y_arr = var_y

        # list of BVLinearGaussianSimulator objects which will simulate the independent processes
        self.bv_simulators = \
            [BVLinearGaussianSimulator(
                  coupling = coupling[i], b_x = b_x[i], b_y = b_y[i], var_x = var_x[i], var_y = var_y[i]) for i in range(n_dim)
            ]

        # attributes
        self.variables = ['X', 'Y']

    def simulate(
            self, 
            time, 
            seed:       int
            ) ->        tuple[np.ndarray, np.ndarray]:
        """
        Simulates a stochastic coupled linear system.
        
        Returns:
        -------
        X : np.ndarray
            time series of variable X 
        Y : np.ndarray
            time series of variable Y
        """
        assert isinstance(time, int)
        assert time > 0
        assert isinstance(seed, int)

        X = np.zeros((time, self.n_dim))
        Y = np.zeros((time, self.n_dim))

        for i in range(self.n_dim):
            x_i, y_i = self.bv_simulators[i].simulate(time, seed+i)
            X[:, i] = x_i.flatten()
            Y[:, i] = y_i.flatten()

        return  X, Y
    
    def analytic_transfer_entropy(
            self, 
            var_from:   str, 
            var_to:     str
            ) ->        float:
        """
        Analytic Transfer Entropy Calculation for the bivariate linear Gaussian model

        TE Y->X for a single system given in Equation 17 in the supplementary materials of [10.1063/5.0053519]. \\
            As channels are independent, TE Y->X for the multivariate system is the sum of the TE Y->X for each independent system.
        TE X->Y is always 0
        
        Parameters:
        ----------
        var_from : str
            The variable transferring information to the variable of interest
        var_to : str
            The variable of interest

        Returns:
        -------
        float
            transfer entropy of the system, given the parameters
        """
        
        assert var_from in self.variables, f"var_from must be in {self.variables}"
        assert var_to in self.variables, f"var_to must be in {self.variables}"
        assert var_from != var_to, "var_from and var_to must be different variables"
    
        if var_from == 'Y' and var_to == 'X':
            TE_Y2X = 0.0
            for i in range(self.n_dim):
                TE_Y2X += self.bv_simulators[i].analytic_transfer_entropy(var_from, var_to)

            return np.round(TE_Y2X, 4)
        
        elif var_from == 'X' and var_to == 'Y':
            return 0.0