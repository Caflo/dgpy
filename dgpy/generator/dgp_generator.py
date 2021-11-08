import numpy as np
import os
from os import path 
import json
from abc import ABC, abstractmethod # for abstract methods

class DGP_Generator(ABC):

    """ An abstract class that provide dataset generation """

    def __init__(self): # filename = params.json path     
        pass

    @abstractmethod
    def generate_dataset(self, params, opt_level='float64'):
        """
        :param params: instance of a class that represents params for DGP (like dgpy.model LR_Params)
        :return: instance of class DGP_Results
        """
        pass


class DGP_Chunk_Generator(ABC):
    """ An abstract class that provide dataset generation chunk by chunk """

    def __init__(self):   
        pass

    @abstractmethod
    def generate_dataset(self, params, saver, chunks, opt_level='float64'):

        """
        You have not to assign this method to a variable, because it doesn't mantain anything in memory
        :param params: instance of LR_Params
        :param saver: child of class Result_Saver. It specifies which saver will be used (for future implementations)
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        """

        pass