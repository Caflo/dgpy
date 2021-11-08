import numpy as np
import os
import pandas as pd
from os import path 
import json
from dgpy.model.dgp_results import DGP_Results
from dgpy.model.lr_params import LR_Params
from dgpy.generator.correlation_matrix_generator import Correlation_Matrix_Generator
from dgpy.generator.dgp_generator import DGP_Generator
from dgpy.persistence.dgp_results_saver import DGP_Results_Saver
from dgpy.persistence.lr_sanity_check import LR_Sanity_Check
import multiprocessing
from multiprocessing import Process
import time
import csv

class Csv_Results_Saver(DGP_Results_Saver):

    """ An implementation of Result_Saver, useful to save various .csv files from generated results """

    def __init__(self, dir_path): # dir_path: percorso della cartella in cui si vogliono salvare i dati
        """
        :param: dir_path: path of the chosen directory where files will be saved
        """
        super().__init__(dir_path)


    def save_all(self, params=None, result=None, feature_analysis=False, compression=False, compr_type=None, verbose=False, multi_proc=False): # salvo i risultati, i parametri e nel caso genero le analisi sulle feature (del Sanity Check)

        """
        :param params: object of class LR_Params
        :param result: object of class DGP_Results
        :param feature_analysis: if true, will save also analysis on the features by feature_analyzer (see dgpy.persistence.feature_analyzer)
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n
        :param verbose: bool, disable if you don't want verbose output
        :param multi_proc: bool, works only if feature_analysis is true. If set to true executes all tasks with multiprocessing (experimental)
        """

            # in dataset.csv gli indici sono 0,0,1,2,3,4 e non 0,1,2,3,4,5 perche pd.Concat() concatena anche gli indici oltre ai dati
            # siccome la Y ha 1 colonna (quindi solo index 0) e la X ne ha un tot allora viene ripetuto lo 0 della X concatenato allo 0 della Y
            # risolvo con dataframe.columns = range(dataframe.shape[1])

        if compression:
            if not compr_type in ['bz', 'gz', 'xz']:
                raise ValueError("Parameter 'compr_type' is not valid")
                sys.exit(1)

        if compression:
            dataset_filename = self.dir_path + "\\dataset.csv" + "." + compr_type
            corr_matr_filename = self.dir_path + "\\corr_matrix.csv" + "." + compr_type
            cov_matr_filename = self.dir_path + "\\cov_matrix.csv" + "." + compr_type
            betas_filename = self.dir_path + "\\betas.csv" + "." + compr_type
            gen_det_filename = self.dir_path + "\\gen_details.csv" + "." + compr_type
        else:
            dataset_filename = self.dir_path + "\\dataset.csv"
            corr_matr_filename = self.dir_path + "\\corr_matrix.csv"
            cov_matr_filename = self.dir_path + "\\cov_matrix.csv"
            betas_filename = self.dir_path + "\\betas.csv"
            gen_det_filename = self.dir_path + "\\gen_details.csv"

        if verbose:
            print("Data will be stored in {}".format(self.dir_path))

        if params != None:
            params.betas.to_csv(betas_filename, header=True, index=False)

        

        if result == None:    
            raise ValueError("Must enter 'result' param") 
        else:
            dataset = pd.concat([result.response, result.features], axis=1, sort=False, copy=False)
            dataset.columns = range(dataset.shape[1]) # resetting indexes
            
            dataset.to_csv(dataset_filename, header=True, index=False)
            
            result.cov_matrix.to_csv(cov_matr_filename, header=True, index=False)
            
            result.corr_matrix.to_csv(corr_matr_filename, header=True, index=False)

            gen_dict = {}
            gen_dict['Start time'] = result.gen_details.start_ts
            gen_dict['End time'] = result.gen_details.end_ts
            gen_dict['Elapsed time'] = result.gen_details.end_ts - result.gen_details.start_ts
            gen_dict['OS'] = result.gen_details.platform_name
            gen_dict['User'] = result.gen_details.user
            df = pd.Series(gen_dict).reset_index()
            df.to_csv(gen_det_filename, header=False, index=False, sep=":")

            if feature_analysis: # creo cartella "Sanity Checks" nel path: <base_path> --> Saves_<dirname> --> 
                sanity_path = self.dir_path + "\\Sanity_Checks"
                if verbose:
                    print("Storing sanity checks in {}".format(sanity_path))
                if not path.exists(sanity_path):
                    os.mkdir(sanity_path)
                # save sanity checks
                fa = LR_Sanity_Check() # feature analyzer
                fa.plot_all(features=result.features, dir_path=sanity_path, verbose=verbose, multi_proc=multi_proc)


    def save_dataframe(self, dataframe, filename, mode='w', index=False, header=True, sep=','):
        """
        :param df: pandas.Dataframe that will be saved
        :param filename: filename where file will be saved
        :param mode: 'w' for writing, 'a' for append. Default 'w'
        :param index: bool, save index of dataframe
        :param header: bool, save header of dataframe
        """

        if path.exists(filename):
            raise ValueError("Path is required")
            sys.exit(1)

        if filename is None:
            raise ValueError("Path is required")
            sys.exit(1)

        filename = self.dir_path + "\\" + filename

        dataframe.to_csv(filename, mode=mode, index=index, header=header, sep=sep)