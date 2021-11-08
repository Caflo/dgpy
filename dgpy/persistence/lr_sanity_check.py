import os
import pandas as pd
from os import path
import multiprocessing
from multiprocessing import Process
import threading
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from dgpy.persistence.dgp_sanity_check import DGP_Sanity_Check
from datetime import datetime
import numpy as np


class LR_Sanity_Check(DGP_Sanity_Check):

    def __init__(self):
        pass


    def plot_value_indip(self, features=None, dir_path=None, verbose=False, return_figs=False): #1) plot each value against the subsequent one

        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features)
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes    
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)

        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        :param verbose: disable if you don't want verbose output
        """
        figs = []

        if dir_path and not path.exists(dir_path):
            raise ValueError("Parameter 'dir_path' is not valid")

        if verbose:
            print("Plotting value indipendency...")

        for (columnName, columnData) in features.iteritems():
            if return_figs:
                fig = plt.figure()
            col_data = columnData.values
            partial_data_1 = col_data[:len(col_data) - 1]
            partial_data_2 = col_data[1:]
            plt.scatter(partial_data_1, partial_data_2)
            plt.title("Independence variable {} Point to point".format(columnName))
            if dir_path:
                plt.savefig(dir_path +"\\IndipP2P_{}".format(columnName))
            if return_figs:
                figs.append(fig)
            
        if verbose:
            print("Done!")

        if return_figs:
            return figs

    def plot_density_hist(self, features=None, dir_path=None, verbose=False, return_figs=False): #2) Histogram of density

        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param verbose: disable if you don't want verbose output
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        """
        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features)
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)


        figs = []

        # if features.dtypes[0] == 'float32' or features.dtypes[0] == 'float16':
        #     raise ValueError("Features DataFrame has dtype = {}. Dtype must be at least 'float64'. If you optimized dataset generation, please turn it to 'false'".format(features.dtypes[0]))
        if dir_path and not path.exists(dir_path):
            raise ValueError("Parameter 'dir_path' is not valid")
            return

        if verbose:
            print("Plotting density histogram...")
        for (columnName, columnData) in features.iteritems():
            if return_figs:
                fig = plt.figure()
            col_data = columnData.values
            plt.hist(col_data,bins=100) # different exception with bins='auto', optimize=True and opt_level=1/2
            plt.title("Distribution variable {}".format(columnName))
            if dir_path:
                plt.savefig(dir_path +"\\Distrib_{}".format(columnName))
            if return_figs:
                figs.append(fig)

        if verbose:
            print("Done!")

        if return_figs:
            return figs

    def plot_autocorr_function(self, features=None, dir_path=None, verbose=False, return_figs=False): #3) Autocorrelation function

        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features)
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)


        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        :param verbose: disable if you don't want verbose output
        """

        if dir_path and not path.exists(dir_path):
            raise ValueError("Parameter 'dir_path' is not valid")

        if verbose:
            print("Plotting autocorrelation function...")

        figs = []

        for (columnName, columnData) in features.iteritems():
            if return_figs:
                fig = plt.figure()
            #(check if each value is correlated to the 20 values ahead)
            col_data = columnData.values
            sm.graphics.tsa.plot_acf(col_data,lags=20)
            plt.title("Autocorrelation variable {}".format(columnName))
            if dir_path:
                plt.savefig(dir_path + "\\Autocorr_{}".format(columnName)) 
            if return_figs:
                figs.append(fig)

        if verbose:
            print("Done!")

        if return_figs:
            return figs


    def plot_p2p_indip(self, features=None, dir_path=None, verbose=False, return_figs=False): #4) Independence between variables 2-2

        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features)
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)


        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        :param verbose: disable if you don't want verbose output
        """
        figs = []

        if dir_path and not path.exists(dir_path):
            raise ValueError("Parameter 'dir_path' is not valid")


        if verbose:
            print("Plotting point to point indipendency...")

        for (columnName, columnData) in features.iteritems():
            for (columnName2, columnData2) in features.iteritems():
                if return_figs:
                    fig = plt.figure()
                col_data = columnData.values
                if columnName != columnName2:
                    col_data2 = columnData2.values
                    plt.scatter(col_data,col_data2)
                    plt.title("Independence of Variables {}, {}".format(columnName, columnName2))
                    if dir_path:
                        plt.savefig(dir_path +"\\Indip_{}_{}".format(columnName, columnName2))
                    if return_figs:
                        figs.append(fig)

        if verbose:
            print("Done!")

        if return_figs:
            return figs


    def plot_spearman_correlation(self, features=None, dir_path=None, verbose=False, return_figs=False):

        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features)
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)



        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        :param verbose: disable if you don't want verbose output
        """

        if dir_path and not path.exists(dir_path):
            raise ValueError("Parameter 'dir_path' is not valid")

        if verbose:
            print("Plotting spearman correlation...")

        corr = features.corr(method="spearman")
        if return_figs:
            fig = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
        plt.title("Matrix of Spearman Correlation")
        if dir_path:
            plt.savefig(dir_path + "\\Correlation")
        

        if verbose:
            print("Done!")

        if return_figs:
            return fig


    def plot_all(self, features=None, dir_path=None, verbose=False, multi_proc=False): # plot all

        if isinstance(features, str): # if value of features is a str will be evaluated as a path
            if path.exists(features): # check if file exists
                if verbose:
                    print("Reading features from {}".format(features))
                features = pd.read_csv(features) # read csv file
                features.drop(features.columns[0], axis=1, inplace=True) # delete Y vector    
                features.columns = range(features.shape[1]) # resetting indexes
            else:
                raise ValueError("features csv file doesn't exist")
                sys.exit(1)


        """
        :param features: pandas Dataframe
        :param dir_path: (optional) directory path where to save files
        :param verbose: disable if you don't want verbose output
        :param return_figs: if set to true will return all the figs. It can consume a lot of memory
        :param multi_proc: set true for multiprocessing mode (experimental)
        """
        start_ts = datetime.now()

        if multi_proc:
            fa_multi_proc = Feature_Analyzer_MultiProc_Executor(features, dir_path, verbose, self)
            fa_multi_proc.run_processes()

        else:
            self.plot_value_indip(features, dir_path, verbose)
            self.plot_density_hist(features, dir_path, verbose)
            self.plot_autocorr_function(features, dir_path, verbose)
            self.plot_p2p_indip(features, dir_path, verbose)
            self.plot_spearman_correlation(features, dir_path, verbose)

        end_ts = datetime.now()

        if verbose:
            print("Analysis time: {}".format(end_ts - start_ts))


class Feature_Analyzer_MultiProc_Executor():
    """
    Class that runs analysis in multiprocessing
    """

    # Il multiprocessing è processi in parallelo, multithreading è thread in parallelo in concorrenza tra loro
    # (perché condividono memoria) nello stesso processo. Dai test i thread risultano incompatibili con la
    # libreria matplotlib che non è thread safe quindi ho optato per multiprocessing. Il multiprocessing
    # ha un costo di creazione maggiore ma è molto più veloce rispetto al multithreading (proprio perché)
    # ci sono processi completamente separati che sfruttano CPU diverse.

    # DA RICORDARE: La ragione per cui nei test il multithreading è leggermente più lento potrebbe essere 
    # perché le CPU sul mio pc sono 4 e mettere più di 4 processi causerebbe la concorrenza negli stessi processi
    # nel prendere le CPU oppure semplicemente il costo di creazione di ogni processo è abbastanza alto, 
    # quindi sarebbe consigliato fare multiprocessing per eseguire in parallelo molte più funzioni

    def __init__(self, features=None, dir_path=None, verbose=False, fa=None):

        
        """
        :param features: pandas Dataframe, it represents the features that will be analyzed
        :param dir_path: str, path where analysis will be saved
        :param verbose: bool, set true if you want verbose output
        :param fa: object of class Feature_Analyzer
        """
        self.features = features
        self.dir_path = dir_path
        self.verbose = verbose

        self.fa = fa

    def run_processes(self):
        """ This method runs parallel processes """
        if self.verbose:
            print("Running 5 processes ...")
            print("Number of CPUs: {}".format(multiprocessing.cpu_count()))
        procs = []

        procs.append(Process(target=self.fa.plot_value_indip, args=(self.features, self.dir_path, self.verbose)))
        procs.append(Process(target=self.fa.plot_density_hist, args=(self.features, self.dir_path, self.verbose)))
        procs.append(Process(target=self.fa.plot_autocorr_function, args=(self.features, self.dir_path, self.verbose)))
        procs.append(Process(target=self.fa.plot_p2p_indip, args=(self.features, self.dir_path, self.verbose)))
        procs.append(Process(target=self.fa.plot_spearman_correlation, args=(self.features, self.dir_path, self.verbose)))

        [proc.start() for proc in procs]
        [proc.join() for proc in procs]
