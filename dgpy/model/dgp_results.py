# Questa classe si occupa di tenere i risultati che si aspettano dalla generazione del dataset

import pandas as pd

class DGP_Results:

    """ A class for DGP Results """

    def __init__(self, features, response, cov_matrix, corr_matrix, gen_details):

        # check params
        self.validate_params(features, response, cov_matrix, corr_matrix, gen_details)

    def validate_params(self, features, response, cov_matrix, corr_matrix, gen_details):

        if features is None:
            raise ValueError("Parameter 'features' is not valid")
        else:
            self.features = features

        if response is None:
            raise ValueError("Parameter 'response' is not valid")
        else:
            self.response = response

        if cov_matrix is None:
            raise ValueError("Parameter 'cov_matrix' is not valid")
        else:
            self.cov_matrix = cov_matrix

        if corr_matrix is None:
            raise ValueError("Parameter 'corr_matrix' is not valid")
        else:
            self.corr_matrix = corr_matrix
        
        if gen_details is None:
            raise ValueError("Parameter 'gen_details' is not valid")
        else:
            self.gen_details = gen_details