from ex_16_4_optimization_problem import *
from osyczka2 import Osyczka2_OptimizationProblem
from typing import List, Dict
from optlib_pymoo_proto import PymooOptimizationAlgorithmMulti, PymooOptimizationAlgorithmSingle
from optlib_jmetalpy_proto import jmetalOptimizationAlgorithmMulti
from optlib_scipy import ScipyOptimizationAlgorithm
import os

# Prepare output directory for writing
out_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
isExist = os.path.exists(out_folder_path)
if not isExist:
    os.makedirs(out_folder_path)

op = EX_16_4_OptimizationProblem('EX_16_4')
#op = Osyczka2_OptimizationProblem('OSY')
#SciPy algorithms
if True:
    opt_ctrl = {}
    op.opt_algorithm = ScipyOptimizationAlgorithm('SLSQP_mi=1000','SLSQP',opt_ctrl)
    if True:
        sol = op.optimize()
        op.print_output()
    else:
        sol = op.optimize_and_write(out_folder_path)

if True:
    pop_size = 100
    num_iter = 50
    max_evaluations = pop_size * num_iter
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.95}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': pop_size, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}
    if False:#ga
        ga_ctrl = {'pop_size': pop_size,'termination': termination}
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('ga_default', 'ga', alg_ctrl=ga_ctrl)
        sol = op.optimize()
        op.print_output()
    if False:#de
        ga_ctrl = {'pop_size': pop_size,'termination': termination}
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('de_default', 'de', alg_ctrl=ga_ctrl)
        sol = op.optimize()
        op.print_output()
    if False:
        brkga_ctrl = {'n_elites': 100, 'n_offsprings': 400,'n_mutants' : 100,'bias' : 0.7,'termination': termination}
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('brkga_default', 'brkga', alg_ctrl=brkga_ctrl)
        sol = op.optimize()
        op.print_output()
if True:#nelder-mead
    termination = ('n_eval', 200)
    nm_ctrl = {'termination': termination}
    op.opt_algorithm = PymooOptimizationAlgorithmSingle('nelder-mead_default', 'nelder-mead', alg_ctrl=nm_ctrl)
    sol = op.optimize()
    op.print_output()