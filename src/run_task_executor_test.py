from ex_16_4_optimization_problem import *
from concurrent.futures import ProcessPoolExecutor,Future
from typing import List, Dict
from jobbase import MultibjectiveOptimizationComparer

def main():
    ops:List[OptimizationProblem] = []
    op = EX_16_4_OptimizationProblem('EX_16_4_COBYLA')
    opt_ctrl = {
        'method': 'COBYLA'}  # ovo je dictionary koji se šalje konstruktoru ScipyOptimizationAlgorithm-a.. to znaci da su ostale postavke defaultne..
    # opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)
    ops.append(op)
    op = EX_16_4_OptimizationProblem('EX_16_4_SLSQP')
    opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)
    ops.append(op)
    job = MultibjectiveOptimizationComparer('test',ops,2)
    job.execute()

if __name__ == '__main__':
    main()