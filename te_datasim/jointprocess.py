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

        z=np.random.normal(0, 1, time+1)
        x=np.random.normal(0, self.rho, time)     
        zp=np.random.normal(0, np.sqrt(1-self.rho*self.rho), time+1)
        y=np.zeros(time+1)
        
        for i in range(time):
            if y[i]<self.lam:
                y[i+1]=z[i+1]
            else:
                y[i+1]=x[i]+zp[i+1]  
                
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