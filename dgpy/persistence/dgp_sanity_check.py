from abc import ABC, abstractmethod
from os import path 

class DGP_Sanity_Check(ABC):

    """ An abstract class for various statistic tests on dataset """

    def __init__(self): # features è il DataFrame delle variabili X1....Xp
        pass

    @abstractmethod
    def plot_value_indip(self, features):
        """ Plot each value against the subsequent one """

        """
        :param features: pandas Dataframe, it represents the features that will be analyzed 
        """

        pass

    @abstractmethod
    def plot_density_hist(self, features):
        """ Histogram of density """
        """
        :param features: pandas Dataframe, it represents the features that will be analyzed 
        """
        pass

    @abstractmethod
    def plot_autocorr_function(self, features):
        """ Autocorrelation function for 20 lags """
        """
        :param features: pandas Dataframe, it represents the features that will be analyzed 
        """
        pass

    @abstractmethod
    def plot_p2p_indip(self, features):
        """ Independence between variables 2-2 """
        """
        :param features: pandas Dataframe, it represents the features that will be analyzed 
        """
        pass

    @abstractmethod
    def plot_spearman_correlation(self, features):
        """ Spearman correlation """
        """
        :param features: pandas Dataframe, it represents the features that will be analyzed 
        """
        pass
