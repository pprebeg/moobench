from ex_16_4_optimization_problem import *
from osyczka2 import Osyczka2_OptimizationProblem
from typing import List, Dict
from optlib_pymoo_proto import PymooOptimizationAlgorithmMulti
from optlib_jmetalpy_proto import jmetalOptimizationAlgorithmMulti
from optlib_scipy import ScipyOptimizationAlgorithm
import os

opt_ctrl = {}
op = EX_16_4_OptimizationProblem('EX_16_4')
op.opt_algorithm = ScipyOptimizationAlgorithm('SLSQP_1','SLSQP',opt_ctrl)
sol = op.optimize_and_write()
print(sol)


#out_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
#isExist = os.path.exists(out_folder_path)
#if not isExist:
#    os.makedirs(out_folder_path)
#sol = op.optimize_and_write(out_folder_path)
