import numpy as np
from scipy.stats import norm

class BVJointProcessSimulator():
    """
    BiVariate Linear Gaussian Simulator
    DAG: X -> Y
    """
    def __init__(self, rho = 0.9, lam = 0.0):
        assert isinstance(rho, float)
        assert isinstance(lam, float)

        # parameters of the linear system
        self.rho = rho
        self.lam = lam

        # attributes
        self.variables = ['X', 'Y']

    def simulate(
            self, 
            time, 
            seed:       int
            ) ->        tuple[np.ndarray, np.ndarray]:
        """
        Simulates the joint process system
        
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

        np.random.seed(seed) # set random seed for reproducibility

        # z=np.random.normal(0, 1, time+1)
        # x=np.random.normal(0, 1, time)     
        # pz = np.sqrt(1-self.rho*self.rho)*z
        # px = self.rho*x
        # y=np.zeros(time+1)
        # p=norm(0, 1).cdf(self.lam)
        # for i in range(time):
        #     y[i+1] = np.average([z[i+1], px[i]+pz[i+1]], weights=[p, 1-p])
                
        # y_ts=y[0:time]
        # Y=y_ts.reshape(-1,1)
    
        # X=x.reshape(-1,1)

        z=np.random.normal(0, 1, time+1)
        x=np.random.normal(0, 1, time)     
        pz = np.sqrt(1-self.rho*self.rho)*z
        px = self.rho*x
        y=np.zeros(time+1)
        
        for i in range(time):
            if y[i]<self.lam:
                y[i+1]=z[i+1]
            else:
                y[i+1]=px[i]+pz[i+1]  
                
        y_ts=y[0:time]
        Y=y_ts.reshape(-1,1)
    
        X=x.reshape(-1,1)
        
        return X, Y
    
    def analytic_transfer_entropy(
            self, 
            var_from:   str, 
            var_to:     str
            ) ->        float:
        """
        Analytic Transfer Entropy Calculation for the bivariate linear Gaussian model

        TE X->Y is given in Section IV of https://doi.org/10.48550/arXiv.1912.07277
        TE Y->X is always 0
        
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
    
        if var_from == 'X' and var_to == 'Y':
            p=norm(0, 1).cdf(self.lam)
            return -(1-p)*0.5*np.log(1-self.rho*self.rho)
        
        elif var_from == 'Y' and var_to == 'X':
            return 0.0
        

class MVJointProcessSimulator():
    """
    MultiVariate Linear Gaussian Simulator
    DAG: X -> Y

    Simply a wrapper for multiple independent BVJointProcessSimulator instances
    however, n_redunant_dim can be specified to add independent dimensions to the system that do not transfer information
    """
    def __init__(self, rho = 0.9, lam = 0.0, n_dim=None, n_redundant_dim=0):
        if all (isinstance(i, float) for i in [rho, lam]):
            assert n_dim is not None, 'n_dim must be specified as the number of independent duplicate dimensions if all parameters are floats'
            assert isinstance(n_dim, int)
            assert n_dim > 0
            assert n_redundant_dim >= 0, 'n_redundant_dim must be a non-negative integer'

            rho = [rho]*n_dim
            lam = [lam]*n_dim
            
        elif all (isinstance(i, list) for i in [rho, lam]):
            assert len(rho) == len(lam), 'rho and lam must have the same length'
            n_dim = len(rho)

        self.n_dim = n_dim - n_redundant_dim
        self.n_redundant_dim = n_redundant_dim

        # parameters of the linear system
        self.rho = rho
        self.lam = lam

        # list of BVJointProcessSimulator instances
        self.jp_simulators = \
            [BVJointProcessSimulator(
                rho=rho[i], lam=lam[i]) for i in range(n_dim)
            ]

        # attributes
        self.variables = ['X', 'Y']

    def simulate(
            self, 
            time, 
            seed:       int
            ) ->        tuple[np.ndarray, np.ndarray]:
        """
        Simulates the joint process system
        
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
            x_i, y_i = self.jp_simulators[i].simulate(time, seed+i)
            X[:, i] = x_i.flatten()
            Y[:, i] = y_i.flatten()

        if self.n_redundant_dim > 0:
            X = np.hstack([X, np.random.normal(0, 1, (time, self.n_redundant_dim))])
            Y = np.hstack([Y, np.random.normal(0, 1, (time, self.n_redundant_dim))])
        
        return X, Y
    
    def analytic_transfer_entropy(
            self, 
            var_from:   str, 
            var_to:     str
            ) ->        float:
        """
        Analytic Transfer Entropy Calculation for the bivariate linear Gaussian model

        TE X->Y is given in Section IV of https://doi.org/10.48550/arXiv.1912.07277. \\
            As channels are independent, TE X->Y for the multivariate system is the sum of the TE X->Y for each independent system.
        TE Y->X is always 0
        
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
    
        if var_from == 'X' and var_to == 'Y':
            TE_Y2X = 0.0
            for i in range(self.n_dim):
                TE_Y2X += self.jp_simulators[i].analytic_transfer_entropy(var_from, var_to)

            return np.round(TE_Y2X, 4)
        
        elif var_from == 'Y' and var_to == 'X':
            return 0.0