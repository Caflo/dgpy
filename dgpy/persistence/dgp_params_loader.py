import numpy as np
import os
from os import path 
import json
from abc import ABC, abstractmethod # for abstract methods

class DGP_Params_Loader(ABC):
    
    """ An abstract class for params loading """

    def __init__(self, filepath):

        """
        :param filepath: param file path
        """

        if not path.exists(filepath): 
            raise ValueError("Parameter 'filepath' is not valid")
        else:
            self.filepath = filepath

    @abstractmethod
    def load(self):

        
        pass