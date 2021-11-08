import numpy as np
import pandas as pd
from scipy.stats import random_correlation
import numba
from numba import jit


class Correlation_Matrix_Generator():

    """ Static class that generates correlation matrix and standard deviations. Useful for many steps that require those generations """


    @staticmethod
    def gen_corr_matrix(seed, n_variables, related_vars, opt_level='float64'):
        
        """ 
        :param n_variables: number of features
        :param related_vars: number of related variables
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'.
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32;\n
        'float16' -> downcast to float16

        Generate a custom correlation matrix: the upper left corner of this matrix gets fulfilled by the partial correlation matrix, obtaining both correlated and incorrelated features
        """

        # eigenvalues (la somma è uguale alla dimensione della matrice)
        eigenvalues = [np.random.uniform(0.3,1.5) for _ in range(max(related_vars-2,2))]
        if len(eigenvalues) == (related_vars-2):
            eigenvalues.append(np.random.uniform((related_vars-sum(eigenvalues))/2.5,
                                                (related_vars-sum(eigenvalues))/1.5))
        if len(eigenvalues) == (related_vars-1):
            eigenvalues.append(related_vars-sum(eigenvalues))
        
        correlation_matrix_part = random_correlation.rvs(eigenvalues) # based on np.random.seed
        
        correlation_matrix_x = np.zeros((n_variables,)*2)
        np.fill_diagonal(correlation_matrix_x, [1 for _ in range(n_variables)])
        correlation_matrix_x[0:related_vars,0:related_vars] = correlation_matrix_part

        return pd.DataFrame(correlation_matrix_x.astype(dtype='float64', copy=True))

    @staticmethod
    def gen_standard_devs(seed, n_variables, related_vars, n_scales, opt_level='float64'):

        """
        :param n_variables: number of features
        :param related_vars: number of related variables
        :param n_scales: number of scales
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'
        
        Notes
        ------
        
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16
        """


        variance_scales = [[5,10],[10,20],[100,200],[1000,2000]]
    
        #Calculate number of unrelated variables
        unrelated_vars = n_variables - related_vars
        
        #n_vars_scale: list with two lists, # of varaibles with the same scale 
        #(at half list you have to repeat (related first, unrelated second))
        n_vars_scale_inner = []
        for n in related_vars,unrelated_vars:
            n1,n2 = divmod(n,n_scales)
            n_vars_scale_inner.append([n1+n2 if pos==0 else n1 for pos in range(n_scales)])
        n_vars_scale = [val for sublist in n_vars_scale_inner for val in sublist]
        
        #Generate vector of standard deviations of the Variables
        stdevs = np.array([np.sqrt(round(np.random.uniform(variance_scales[pos][0],
                                                        variance_scales[pos][1]),1)) 
        for pos,rep in zip(list(range(int(len(n_vars_scale)/2)))*2,n_vars_scale) for _ in range(rep)])
        stdevs = np.array(stdevs)[np.newaxis]
        return stdevs.astype(dtype=opt_level, copy=False)      