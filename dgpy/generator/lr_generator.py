import numpy as np
import os
import pandas as pd
from os import path
import json
from dgpy.model.dgp_results import DGP_Results
from dgpy.model.lr_params import LR_Params
from dgpy.model.gen_details import Generation_Details
from dgpy.generator.correlation_matrix_generator import Correlation_Matrix_Generator
from dgpy.generator.dgp_generator import DGP_Generator, DGP_Chunk_Generator
from dgpy.persistence.dgp_results_saver import DGP_Results_Saver
from datetime import datetime

class LR_Generator(DGP_Generator): # Linear Regression Generator

    """
    Implementation of DGP_Generator, used for Linear Regression.
    Its functions generate dataset with optimization level, for memory saving
    see dgpy.lr_generator.LR_Chunk_Generator for dataset generation chunk by chunk
    """

    def __init__(self):
        super().__init__()

    def generate_dataset(self, params, verbose=False, opt_level='float64'):

        # LR GENERATING PROCESS
        """
        :param params: instance of LR_Params
        :param verbose: verbose output
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
                'float64' -> default (no optimization)\n
                'float32' -> downcast to float32\n
                'float16' -> downcast to float16\n

        :return: instance of class DGP_Results
        """


        if opt_level != 'float32' and opt_level != 'float16' and opt_level != 'float64':
            raise ValueError("Parameter 'opt_level' is not valid")
            sys.exit(1)

        start_ts = datetime.now()

        if verbose:
            if opt_level == 'float32':
                print("Generating dataset with optimization level: {}".format(opt_level))
            elif opt_level == 'float16':
                print("Generating dataset with optimization level: {}".format(opt_level))
                print("Warning: possible loss of arithmetic result precision")
            print("Start time: {}".format(start_ts))


        # Ad ogni generazione reinizializzo la randomizzazione
        # Questo perché se invoco la stessa funzione ad esempio 2 volte nel codice cliente
        # i risultati verrebbero sballati nella seconda in quando viene modificato il random state
        params.initialize_randomization()

        stdevs = Correlation_Matrix_Generator.gen_standard_devs(params.seed, params.n_variables, params.related_vars, params.n_scales, opt_level=opt_level)
        correlation_matrix_x = Correlation_Matrix_Generator.gen_corr_matrix(params.seed, params.n_variables, params.related_vars, opt_level=opt_level)

        # Creo la Covariance Matrix a partire dalla correlation matrix e dalle stdevs
        covariance_matrix = np.dot(stdevs.T, stdevs) * correlation_matrix_x

        x_state = np.random.RandomState(params.seed)
        x_state.set_state(np.random.get_state())
        # se mi salvo lo stato (ad esempio x_state) devo essere nello stato giusto per poter generare gli stessi numeri.
        # Inserendo np.random.RandomState(params.seed) resetto il generatore, non voglio questo.

        y_state = np.random.RandomState(params.seed)
        y_state.set_state(np.random.get_state())

        # Genero dati da Normale Multivariata (X rv)
        X = pd.DataFrame(x_state.multivariate_normal(mean=params.means, cov=covariance_matrix,\
                                                         size=params.n_points).astype(dtype=opt_level))
        mean_y = pd.Series(params.betas[0] + np.dot(X, params.betas[1:]))
        stdev_y = np.sqrt(params.error_variance)


        Y = pd.Series(y_state.normal(mean_y, stdev_y).astype(dtype=opt_level))

        end_ts = datetime.now()

        if verbose:
            print("End time: {}".format(end_ts))
            print("Generating process time: {}".format(end_ts - start_ts))
            x_usage = X.memory_usage(deep=True).sum()
            y_usage = Y.memory_usage(deep=True)
            print("Total memory usage of [Y, X]: {}\n".format(ByteFormatter.format_bytes(x_usage + y_usage)))


        gen_details = Generation_Details(start_ts, end_ts)
        result = DGP_Results(X, Y, covariance_matrix, correlation_matrix_x, gen_details)

        # dovrei tornare indietro con lo stato perché altrimenti nel codice cliente
        # se invoco i due generatori uno dopo l'altro non producono gli stessi dati

        return result





class LR_Chunk_Generator(DGP_Chunk_Generator):

    """
    Implementation of DGP_Generator, used for Linear Regression.
    Its functions generate dataset chunk by chunk with optimization level, for memory saving
    """

    def generate_dataset(self, params, saver, chunks=1, verbose=False, opt_level='float64', compression=False, compr_type=None):
        """
        :param params: instance of LR_Params
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        :param verbose: verbose output
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n\t
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n\t

        You may not assign this method to a variable, because it doesn't mantain anything in memory
        """

        # Ad ogni generazione reinizializzo la randomizzazione
        # Questo perché se invoco la stessa funzione ad esempio 2 volte nel codice cliente
        # i risultati verrebbero sballati nella seconda in quando viene modificato il random state
        params.initialize_randomization()

        if compression:
            if not compr_type in ['bz', 'gz', 'xz']:
                raise ValueError("Parameter 'compr_type' is not valid")
                sys.exit(1)

        if params.n_points % chunks != 0:
            raise ValueError("Parameter 'chunks' is not valid. param 'n_points' must be divisible by param 'chunks'")
            sys.exit(1)

        if chunks > params.n_points or chunks <= 0:
            raise ValueError("Parameter 'chunks' is not valid")
            sys.exit(1)

        if compression:
            dataset_filename = "dataset.csv" + "." + compr_type
            corr_matr_filename = "corr_matrix.csv" + "." + compr_type
            cov_matr_filename = "cov_matrix.csv" + "." + compr_type
            betas_filename = "betas.csv" + "." + compr_type
            gen_det_filename = "gen_details.csv" + "." + compr_type
        else:
            dataset_filename = "dataset.csv"
            corr_matr_filename = "corr_matrix.csv"
            cov_matr_filename = "cov_matrix.csv"
            betas_filename = "betas.csv"
            gen_det_filename = "gen_details.csv"


        stdevs = Correlation_Matrix_Generator.gen_standard_devs(params.seed, params.n_variables, params.related_vars, params.n_scales, opt_level=opt_level)
        correlation_matrix_x = Correlation_Matrix_Generator.gen_corr_matrix(params.seed, params.n_variables, params.related_vars, opt_level=opt_level)
        covariance_matrix = np.dot(stdevs.T, stdevs) * correlation_matrix_x

        saver.save_dataframe(correlation_matrix_x, corr_matr_filename)
        saver.save_dataframe(covariance_matrix, cov_matr_filename)
        saver.save_dataframe(params.betas, betas_filename)


        chunksize = int(params.n_points / chunks)
        rest = params.n_points % chunks
        stdev_y = np.sqrt(params.error_variance)
        start_ts = datetime.now() # start counter

        # Problematica riguardo i randomState: mettere nello stesso ciclo random_normal e random.multivariate_normal causava un cambio dello stato da parte
        # di np.random_normal e al ciclo dopo i dati venivano 'sballati', ovvero sempre randomizzati ma con una randomizzazione diversa,
        # quindi i dati differivano da quelli generati dal controller LR_Generator
        # in breve le due funzioni condividono lo stesso np.random State



        # SOLUZIONE 1: fare due cicli separati, uno per X e uno per Y, che sdoppia il tempo di generazione, ma per lo meno genera i dati giusti (non so nemmeno se si possa fare)

        # SOLUZIONE 2: riprendermi lo stesso stato dentro il ciclo, sdoppiando i RandomState -> ottimizzata e giusta, uso questa



        # Dataset generation


        # se mi salvo lo stato (ad esempio x_state) devo essere nello stato giusto per poter generare gli stessi numeri.
        # Inserendo np.random.RandomState(params.seed) resetto il generatore, ciò non basta. Devo anche settare lo stato
                
        x_state = np.random.RandomState(params.seed)
        x_state.set_state(np.random.get_state())

        y_state = np.random.RandomState(params.seed)
        y_state.set_state(np.random.get_state())
        for i in range(0, chunks):


            X = pd.DataFrame(x_state.multivariate_normal(mean=params.means, cov=covariance_matrix, size=chunksize).astype(dtype=opt_level))
            mean_y = pd.Series(params.betas[0] + np.dot(X, params.betas[1:])).astype(dtype=opt_level)
            Y = pd.Series(y_state.normal(mean_y, stdev_y).astype(dtype=opt_level))
            # concat dataframes to form a matrix [Y, X]
            dataset = pd.concat([Y, X], axis=1, sort=False)
            dataset.columns = range(dataset.shape[1]) # resetting indexes
            if i == 0:
                saver.save_dataframe(dataset, dataset_filename, mode='w', header=True, index=False)
            else:
                saver.save_dataframe(dataset, dataset_filename, mode='a', header=False, index=False)

        

        end_ts = datetime.now() # stop counter

        if verbose:
            # print("Number of iterations: {} + {}".format(chunks, rest))
            print("Number of iterations: {}".format(chunks))
            print("Memory usage of [Y, X] (per chunk): {}".format(ByteFormatter.format_bytes(dataset.memory_usage(deep=True).sum())))
            print("Dataset generation time took {}".format(end_ts - start_ts))

        gen_details = Generation_Details(start_ts, end_ts)
        gen_dict = {}
        gen_dict['Start time'] = gen_details.start_ts
        gen_dict['End time'] = gen_details.end_ts
        gen_dict['Elapsed time'] = gen_details.end_ts - gen_details.start_ts
        gen_dict['OS'] = gen_details.platform_name
        gen_dict['User'] = gen_details.user
        gen_df = pd.Series(gen_dict).reset_index()
        saver.save_dataframe(gen_df, gen_det_filename, header=False, index=False, sep=":")

        # dovrei tornare indietro con lo stato perché altrimenti nel codice cliente
        # se invoco i due generatori uno dopo l'altro non producono gli stessi dati



class ByteFormatter:

    @staticmethod
    def format_bytes(B): # format bytes in a human readable form
        B = float(B)
        KB = float(1024)
        MB = float(KB ** 2) # 1,048,576
        GB = float(KB ** 3) # 1,073,741,824
        TB = float(KB ** 4) # 1,099,511,627,776

        if B < KB:
            return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Bytes')
        elif KB <= B < MB:
            return '{0:.2f} KB'.format(B/KB)
        elif MB <= B < GB:
            return '{0:.2f} MB'.format(B/MB)
        elif GB <= B < TB:
            return '{0:.2f} GB'.format(B/GB)
        elif TB <= B:
            return '{0:.2f} TB'.format(B/TB)