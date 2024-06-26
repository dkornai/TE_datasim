import numpy as np

class BVLinearGaussianSimulator():
    """
    BiVariate Linear Gaussian Simulator
    DAG: Y -> X
    """
    def __init__(self, coupling = 0.5, b_x = 0.8, b_y = 0.4, var_x = 0.2, var_y = 0.2):
        assert isinstance(coupling, float)
        assert isinstance(b_x, float)
        assert isinstance(b_y, float)
        assert isinstance(var_x, float)
        assert isinstance(var_y, float)
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