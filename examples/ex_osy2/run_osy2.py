from moobench.optbase import OptimizationProblem
from ex_osy2 import OSY2_OptProb
from moobench.optlib_pymoo_proto import PymooOptimizationAlgorithmMulti

import os

# Prepare output directory for writing
out_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
isExist = os.path.exists(out_folder_path)
if not isExist:
    os.makedirs(out_folder_path)

op = OSY2_OptProb('OSY')
pop_size = 100
num_iter = 100
max_evaluations = pop_size * num_iter
termination = ('n_eval', max_evaluations)
default_ctrl = {'pop_size': pop_size, 'termination': termination}
if True:  # nsga2
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('nsga2_default', 'nsga2', alg_ctrl=default_ctrl)
    # sol = op.optimize()
    op.optimize_and_write(out_folder_path)
    op.print_output()
