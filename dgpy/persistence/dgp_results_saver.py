import numpy as np
import os
from os import path 
import json
from abc import ABC, abstractmethod # for abstract methods
from datetime import datetime

class DGP_Results_Saver(ABC):
    
    def __init__(self, dir_path):
        if not path.exists(dir_path):
            raise ValueError("Parameter 'base_dir' is not valid")
        else:
            self.dir_path = dir_path

    @abstractmethod
    def save_all(self, params=None, result=None):
        pass

    @abstractmethod
    def save_dataframe(self, dataframe, filename):
        """
        :param dataframe: pandas.Dataframe that will be saved
        :param filename: filename where file will be saved
        """
        pass