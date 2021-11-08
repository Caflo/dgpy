import numpy as np
import os
import pandas as pd
from os import path 
import json
from dgpy.model.dgp_results import DGP_Results
from dgpy.model.lr_params import LR_Params
from dgpy.generator.correlation_matrix_generator import Correlation_Matrix_Generator
from dgpy.persistence.dgp_params_loader import DGP_Params_Loader

class Json_Params_Loader(DGP_Params_Loader):

    """ An implementation for params loading from JSON files """

    def __init__(self, filepath):
        """
        :param filepath: str, the filename from where this loader take params (absolute path)
        """
        super().__init__(filepath)

    def load(self, param_type='lr'):
        """
        :param param_type: str, represents the param type (LR_Params or future implementations of DGP_Params). It's used so the reader will return the correct instance of Params. Default 'lr'\n
        This function read a filename chosen in the constructor Json_Params_Loader and uses json methods to load parameters. 
        In this version there's only Linear Regression as param_type, so only LR_Params object will be created. 
        In future versions it will expand 
        """

        # Per implementazioni di regressioni future bisognerà controllare che param_type sia all'interno di una lista già specificata (tanto le regressioni sono note, siccome sono cose matematiche)
        # Basta introdurre questo pezzo di codice:

        # if not param_type in ['lr', 'log', 'other_regression_type'] # e così via.. 
        #     raise ValueError("param_type is not valid. See docs for allowed param_types") # da includere nelle docs

        # Con implementazioni future di DGP_Params sarà necessario inserire nuove casistiche per restituire l'oggetto giusto.
        # Ho scelto così siccome altrimenti avrei avuto tantissime classi, ad esempio un Json_LR_Loader, un Json_Logistic_Loader ecc... (in futuro magari)
        # e le classi sarebbero diventate troppe

        # Quindi ciò che posso fare è leggere tutte le coppie chiave-valore (che andranno in p_dict) e costruire correttamente l'oggetto in base a param_type

        with open(self.filepath, 'r') as f:
            p_dict = json.load(f)

        # construct object
        
        # uso dict.get() così se non trova un campo mi mette automaticamente None
        # nel costruttore di LR_Params ho che se trova None in alcuni valori randomizza, quindi non c'è problema in caso p_dict.get("seed") restituisca None
        # Molto comodo, così passo tutto automaticamente
        if param_type == "lr":
            params = LR_Params(p_dict.get("seed"), p_dict.get("n_variables"), p_dict.get("error_variance"), \
                p_dict.get("n_points"), p_dict.get("related_vars"), p_dict.get("n_scales"), p_dict.get("requested_mean"), \
                    p_dict.get("betas"))
        # elif param_type == 'log':
            # ...
        return params