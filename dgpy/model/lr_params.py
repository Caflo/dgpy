import numpy as np
import pandas as pd

class LR_Params():

    """ Params used for Linear Regression Data Generating Process """
    
    def __init__(self, seed=None, n_variables=None, error_variance=None, \
        n_points=None, related_vars=None, n_scales=None, requested_mean=None, betas=None):

        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_mean: int, mean of vector Y
        :param betas: ndarray. If not set, will be randomized
        """
        
       
        self.validate_params_and_create_obj(seed, n_variables, error_variance, \
            n_points, related_vars, n_scales, requested_mean, betas)


    def validate_params_and_create_obj(self, seed=None, n_variables=None, error_variance=None, n_points=None, \
        related_vars=None, n_scales=None, requested_mean=None, betas=None, gen_x_means=True):

        if seed is None: # seed is randomized
            # se il seed viene generato non e' possibile ricavarselo
            # è possibile invece riprendersi lo stato del seed:
            # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
            self.seed = np.random.seed()
        else:
            self.seed = seed

        # Set the seed
        self.initialize_randomization()
        
        random_copy = np.random.RandomState(self.seed) # mi faccio una copia, non voglio mai alterare lo stato globale

        if n_variables is None or n_variables <= 0:
            raise ValueError("Parameter 'n_variables' is not valid")
            sys.exit(1)
        else:
            self.n_variables = n_variables

        if error_variance is None: 
            raise ValueError("Parameter 'error_variance' is not valid")
            sys.exit(1)
        else:
            self.error_variance = error_variance   

        if n_points is None or n_points <= 0: 
            raise ValueError("Parameter 'n_points' is not valid")
            sys.exit(1)
        else:
            self.n_points = n_points

        if related_vars is None or related_vars <= 0: 
            raise ValueError("Parameter 'related_vars' is not valid")
            sys.exit(1)
        else:
            self.related_vars = related_vars      

        if n_scales is None or related_vars <= 0:
            raise ValueError("Parameter 'n_scales' is not valid")
            sys.exit(1)
        else:
            self.n_scales = n_scales

        # requested_mean represents the X mean
        if requested_mean is None: 
            raise ValueError("Parameter 'requested_mean' is not valid")
            sys.exit(1)
        else:
            self.requested_mean = requested_mean

        # gen X means if true
        self.means = [round(random_copy.uniform(0, 3), 1) for _ in range(n_variables)]
        self.means = list(np.multiply(self.means, [[-1, 1][random_copy.randint(2)] for _ in range(len(self.means))]))

        if betas is None: # is randomized
            # Generate vector of coefficients (random[0,3]), randomly switch some to negative
            betas = [round(random_copy.uniform(-2, 2), 1) for _ in range(n_variables)]

            # Voglio controllare il valore medio (solo la parte deterministica X*beta) nel nostro campione.
            intercept_value = round(requested_mean - sum([a * c for a, c in zip(betas, self.means[:len(betas)])]), 2)

            betas.insert(0, intercept_value)
        
        self.betas = pd.Series(betas) # create pandas object


    def initialize_randomization(self):
        np.random.RandomState(self.seed) # uso il seed per generare il primo stato. Gli altri saranno generati automaticamente
        np.random.seed(self.seed)