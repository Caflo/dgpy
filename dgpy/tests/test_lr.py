import unittest
from unittest import TestCase
import pandas as pd
import numpy as np
import numpy.testing as nt
import sys
import os

from dgpy.generator.lr_generator import LR_Generator, LR_Chunk_Generator
from dgpy.persistence.json_loader import Json_Params_Loader
from pandas.testing import assert_frame_equal, assert_series_equal
from dgpy.model.lr_params import LR_Params
from dgpy.persistence.csv_saver import Csv_Results_Saver

class LR_Test(TestCase):

    def setUp(self):
        print("\n\n\n--------------- SETUP  -------------------")
        print("Starting LR_Test...")

        self.filename = os.path.join(os.path.dirname(__file__), "examples", "example_params.json")
        print(self.filename)
        self.base_dir = os.path.join(os.path.dirname(__file__), "results") # dove salverò i .csv

        # caricamento dei parametri
        self.json_loader = Json_Params_Loader(self.filename)
        self.params = self.json_loader.load()

        # creazioni dei generatori e del saver che riutilizzeremo per i test
        self.generator = LR_Generator()
        self.generator_chunk = LR_Chunk_Generator()
        # self.csv_saver = Csv_Results_Saver(self.base_dir)

        """
        I test vengono eseguiti in ordine alfabetico, perciò metto un numero
        """
    
    # def test_01_generated_dataset_without_opt(self):

    #     print("TESTING GENERATED DATASET WITHOUT OPTIMIZATION ---------------------")


    #     # example_dataset è il dataset di esempio, dovrà essere confrontato con quello generato
    #     # example_dataset è stato creato da un LR_Regressor con i seguenti parametri:

    #     """
    #         Si fa il test su un dataset di esempio, generato con determinati parametri
    #         e vedere se i risultati tornano utilizzando lo stesso seed
    #     """

    #     example_dataset_no_opt = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_dataset.csv"))
    #     example_betas = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_betas.csv"))
    #     example_corr_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_corr_matrix.csv"))
    #     example_cov_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_cov_matrix.csv"))

    #     # must convert ample from DataFrame to Series for checking equality
    #     example_betas = example_betas.iloc[:, 0]

    #     results = self.generator.generate_dataset(self.params, verbose=True)

    #     dataset_no_opt = pd.concat([results.response, results.features], axis=1, sort=False)

      

    #     y1 = example_dataset_no_opt.iloc[:, 0].values
    #     y2 = dataset_no_opt.iloc[:, 0].values

    #     # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
    #     dataset_no_opt.columns = range(dataset_no_opt.shape[1])
    #     example_dataset_no_opt.columns = range(example_dataset_no_opt.shape[1])

    #     example_betas.columns = range(example_betas.shape[0])
    #     self.params.betas.columns = range(self.params.betas.shape[0])

    #     example_corr_matrix.columns = range(example_corr_matrix.shape[1])
    #     results.corr_matrix.columns = range(results.corr_matrix.shape[1])

    #     example_cov_matrix.columns = range(example_cov_matrix.shape[1])
    #     results.cov_matrix.columns = range(results.cov_matrix.shape[1])

    #     # se decommento questo il test fallisce
    #     # dataset.at[0, 2] = 304 


    #     """
    #         A causa di alcune cifre dopo la virgola, alcuni numeri potrebbero non risultare uguali.
    #         Per questo uso la funione np.allclose() che controlla che i numeri siano 'circa' uguali
    #         Per vederlo chiaramente, basta decommentare questo pezzo di codice

    #         print(arr1) # array of dataset 1
    #         print(arr2) # array of dataset 2

    #         i = 0
    #         while i < arr1.size:
    #             if arr1[i] == arr2[i]:
    #                 print("TRUE")
    #             else:
    #                 print("FALSE")
    #              i += 1
    #     """
        
    #     self.assertTrue(np.allclose(y1, y2))
      
    #     assert_frame_equal(dataset_no_opt, example_dataset_no_opt)
    #     assert_series_equal(example_betas, self.params.betas, check_names=False)
    #     assert_frame_equal(example_corr_matrix, results.corr_matrix)
    #     assert_frame_equal(example_cov_matrix, results.cov_matrix)


    # def test_02_generated_dataset_with_opt_level_1(self):

    #     print("TESTING GENERATED DATASET WITH OPTIMIZATION LEVEL 1 (FLOAT 32) ---------------------")

    #     example_dataset_opt_1 = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_dataset.csv"))
    #     example_betas = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_betas.csv"))
    #     example_corr_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples" ,"example_corr_matrix.csv"))
    #     example_cov_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_cov_matrix.csv"))

    #     # forcing to convert dataset to float32 (.astype() doesn't work)
    #     floats = example_dataset_opt_1.select_dtypes(include=['float64']).columns.tolist()
    #     example_dataset_opt_1[floats] = example_dataset_opt_1[floats].apply(pd.to_numeric, downcast='float')


    #     # must convert ample from DataFrame to Series for checking equality
    #     example_betas = example_betas.iloc[:, 0]

    #     results = self.generator.generate_dataset(self.params, verbose=True, opt_level='float32')

    #     dataset_opt_1 = pd.concat([results.response, results.features], axis=1, sort=False)

    #     # dataset_opt_2.info(verbose=True)


    #     # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
    #     dataset_opt_1.columns = range(dataset_opt_1.shape[1])
    #     example_dataset_opt_1.columns = range(example_dataset_opt_1.shape[1])

    #     example_betas.columns = range(example_betas.shape[0])
    #     self.params.betas.columns = range(self.params.betas.shape[0])

    #     example_corr_matrix.columns = range(example_corr_matrix.shape[1])
    #     results.corr_matrix.columns = range(results.corr_matrix.shape[1])

    #     example_cov_matrix.columns = range(example_cov_matrix.shape[1])
    #     results.cov_matrix.columns = range(results.cov_matrix.shape[1])

    #     """
    #         A causa di alcune cifre dopo la virgola, alcuni numeri potrebbero non risultare uguali.
    #         Per questo uso la funione np.allclose() che controlla che i numeri siano 'circa' uguali
    #         Per vederlo chiaramente, basta decommentare questo pezzo di codice

    #         print(arr1)
    #         print(arr2)

    #         i = 0
    #         while i < arr1.size:
    #             if arr1[i] == arr2[i]:
    #                 print("TRUE")
    #             else:
    #                 print("FALSE")
    #              i += 1
    #     """

    #     y1 = example_dataset_opt_1.iloc[:, 0].values
    #     y2 = dataset_opt_1.iloc[:, 0].values
        
    #     self.assertTrue(np.allclose(y1, y2))
    #     self.assertTrue(np.allclose(dataset_opt_1.values, example_dataset_opt_1.values, atol=0.1))       

    
    # def test_03_generated_dataset_with_opt_level_2(self):

    #     print("TESTING GENERATED DATASET WITH OPTIMIZATION LEVEL 2 (FLOAT 16) ---------------------")


    #     # example_dataset è il dataset di esempio, dovrà essere confrontato con quello generato
    #     # example_dataset è stato creato da un LR_Regressor con i seguenti parametri:

    #     seed = 22
    #     n_variables = 5
    #     error_variance = 5
    #     n_points = 50
    #     related_vars = 3
    #     n_scales = 1
    #     requested_mean = 0
    #     betas = [10.22, 1.80, -0.70, 0.40, 1.70, 0.70]

    #     test_params =  LR_Params(seed = seed, n_variables = n_variables, error_variance = error_variance,\
    #                                 n_points = n_points, related_vars = related_vars, n_scales = n_scales,\
    #                                     requested_mean = requested_mean, betas = betas)

    #     """
    #         Si fa il test su un dataset di esempio, generato con determinati parametri
    #         e vedere se i risultati tornano utilizzando lo stesso seed
    #     """

        
    #     example_dataset_opt_2 = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_dataset.csv"))
    #     example_betas = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_betas.csv"))
    #     example_corr_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_corr_matrix.csv"))
    #     example_cov_matrix = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_cov_matrix.csv"))

    #     # convert dataset to float16
    #     example_dataset_opt_2 = pd.DataFrame(example_dataset_opt_2.values.astype(dtype='float16', copy=False))


    #     # must convert ample from DataFrame to Series for checking equality
    #     example_betas = example_betas.iloc[:, 0]

    #     results = self.generator.generate_dataset(self.params, verbose=True, opt_level='float16')

    #     dataset_opt_2 = pd.concat([results.response, results.features], axis=1, sort=False)

    #     # dataset_opt_2.info(verbose=True)


    #     # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
    #     dataset_opt_2.columns = range(dataset_opt_2.shape[1])
    #     example_dataset_opt_2.columns = range(example_dataset_opt_2.shape[1])

    #     example_betas.columns = range(example_betas.shape[0])
    #     self.params.betas.columns = range(self.params.betas.shape[0])

    #     example_corr_matrix.columns = range(example_corr_matrix.shape[1])
    #     results.corr_matrix.columns = range(results.corr_matrix.shape[1])

    #     example_cov_matrix.columns = range(example_cov_matrix.shape[1])
    #     results.cov_matrix.columns = range(results.cov_matrix.shape[1])

    #     """
    #         A causa di alcune cifre dopo la virgola, alcuni numeri potrebbero non risultare uguali.
    #         Per questo uso la funione np.allclose() che controlla che i numeri siano 'circa' uguali
    #         Per vederlo chiaramente, basta decommentare questo pezzo di codice

    #         print(arr1)
    #         print(arr2)

    #         i = 0
    #         while i < arr1.size:
    #             if arr1[i] == arr2[i]:
    #                 print("TRUE")
    #             else:
    #                 print("FALSE")
    #              i += 1
    #     """

    #     y1 = example_dataset_opt_2.iloc[:, 0].values
    #     y2 = dataset_opt_2.iloc[:, 0].values
    #     self.assertTrue(np.allclose(y1, y2, atol=0.3)) # metto una tolleranza più alta perché l'approssimazione è peggiore (toglierlo per vedere come fallisce)
    #     self.assertTrue(np.allclose(dataset_opt_2.values, example_dataset_opt_2.values, atol=0.1)) # stessa cosa


    def test_04_generated_dataset_chunk(self):

        print("TESTING CHUNK BY CHUNK GENERATED DATASET WITHOUT OPTIMIZATION ---------------------")

        chunks = 50

        saver = Csv_Results_Saver(os.path.join(os.path.dirname(__file__), "results_chunk"))

        example_dataset_no_opt = pd.read_csv(os.path.join(os.path.dirname(__file__), "examples", "example_dataset.csv"))
    
        self.generator_chunk.generate_dataset(self.params, saver, chunks=chunks, opt_level='float16', verbose=True)

        dataset_no_opt = pd.read_csv(os.path.join(os.path.dirname(__file__), "results_chunk", "dataset.csv"))

      

        y1 = example_dataset_no_opt.iloc[:, 0].values
        y2 = dataset_no_opt.iloc[:, 0].values

        # Reshape delle colonne delle matrici (mi serve siccome assert_frame controlla anche i tipi delle colonne e questo non mi serve)
        dataset_no_opt.columns = range(dataset_no_opt.shape[1])
        example_dataset_no_opt.columns = range(example_dataset_no_opt.shape[1])

        self.assertTrue(np.allclose(y1, y2))
        self.assertTrue(np.allclose(dataset_no_opt, example_dataset_no_opt)) # senza atol va bene perche in questo caso è stato generato con float64

      
if __name__ == '__main__':
    unittest.main()