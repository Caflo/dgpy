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

class LR_Load_Test(TestCase):        

    def test_01_json_params_load(self):

        self.filename = os.path.join(os.path.dirname(__file__), "examples", "example_params.json") # prendo un file che ha gli stessi parametri specificati sotto

        # json loader
        self.test_params = Json_Params_Loader(self.filename).load()

        seed = 22
        n_variables = 5
        error_variance = 5
        n_points = 50
        related_vars = 3
        n_scales = 1
        requested_mean = 0
        betas = [10.22, 1.80, -0.70, 0.40, 1.70, 0.70]

        prms = LR_Params(seed=seed, n_variables=n_variables, error_variance=error_variance, n_points=n_points, 
                        related_vars=related_vars, n_scales=n_scales, requested_mean=requested_mean, betas=betas)
       
        self.assertTrue(prms.seed == self.test_params.seed)
        self.assertTrue(prms.n_variables == self.test_params.n_variables)
        self.assertTrue(prms.error_variance == self.test_params.error_variance)
        self.assertTrue(prms.n_points == self.test_params.n_points)
        self.assertTrue(prms.related_vars == self.test_params.related_vars)
        self.assertTrue(prms.n_scales == self.test_params.n_scales)
        self.assertTrue(prms.requested_mean == self.test_params.requested_mean)
        nt.assert_array_equal(prms.betas.values, self.test_params.betas.values)

                    
