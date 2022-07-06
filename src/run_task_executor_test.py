from ex_16_4_optimization_problem import *
from Frame_problem import Analiza_okvira_OptimizationProblem
from osyczka2 import Osyczka2_OptimizationProblem
from concurrent.futures import ProcessPoolExecutor,Future
from typing import List, Dict
from jobbase import MultibjectiveOptimizationComparer,MultibjectiveOptimizationComparerFromWrittenResults
from optlib_pymoo_proto import PymooOptimizationAlgorithmMulti
from optlib_jmetalpy_proto import jmetalOptimizationAlgorithmMulti
from optlib_scipy import ScipyOptimizationAlgorithm
import os


def load_pareto_front(op:OptimizationProblem, ref_out_path:str):
    ref_pareto_front = None
    ref_output = op.load_opt_output(ref_out_path)
    if ref_output is not None and isinstance(ref_output,MultiobjectiveOptimizationOutput):
        ref_pareto_front =  ref_output.solutions
    return ref_pareto_front

def opttest_ref_front(name,max_workers,out_folder_path):
    ops:List[OptimizationProblem] = []
    op = Analiza_okvira_OptimizationProblem('Ponton')
    pop_size = 500
    mutation = {'name':'real_pm', 'eta':20, 'prob': 1/op.num_var+0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check

##    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
##    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    
    termination = ('time', '02:00:00')
##    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size,'n_offsprings':10, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1','nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    #job
    job = MultibjectiveOptimizationComparer(name,max_workers,out_folder_path,None)
    job.execute()


def opttest_osy(name,max_workers,out_folder_path):
    ops: List[OptimizationProblem] = []
    pop_size = 100
    num_iter = 20
    max_evaluations = pop_size * num_iter
    #1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    #2
    op = Osyczka2_OptimizationProblem('OSY')

    mutation={'name':'real_pm', 'eta':80, 'prob': 0.9}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':1.0}  # Check Zanimljivo! doslovno potrebno ponovno instancirati, jer nekako drži referencu na ovaj globalni dictionary. .pop() metoda izbije key-value par u ovom globalnom rijecniku!
    termination = ('n_eval', max_evaluations)

    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_2','nsga2', alg_ctrl=alg_ctrl)

    ops.append(op)

    #3
    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check    crossover_obj = get_crossover('real_sbx', eta=25, prob=0.95)  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':1.0}  # Check
    termination = ('n_eval', max_evaluations)

    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_3', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    #4
    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'real_pm', 'eta':80, 'prob': 0.1}  # Check    crossover_obj = get_crossover('real_sbx', eta=25, prob=1.0)  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':1.0}  # Check
    termination = ('n_eval', max_evaluations)

    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_4', 'nsga2',alg_ctrl=alg_ctrl)
    ops.append(op)

    # 1

    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'polynomial', 'distribution_index':0.20, 'probability': 0.9}  # Check
    crossover = {'name':'sbx', 'distribution_index':20, 'probability':1.0}  # Check

    selection = {'name':'bts'}
    termination = {'name':'n_eval','max_evaluations':max_evaluations}
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size':pop_size, 'selection':selection, 'mutation': mutation,'crossover': crossover, 'termination':termination}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_1', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    # 2
    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'polynomial', 'distribution_index':0.80, 'probability': 0.9}  # Check
    crossover = {'name':'sbx', 'distribution_index':20, 'probability':1.0}  # Check
    selection = {'name':'bts'}
    termination = {'name':'n_eval','max_evaluations':max_evaluations}
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size':pop_size, 'selection':selection, 'mutation': mutation,'crossover': crossover, 'termination':termination}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_2', 'nsga2',alg_ctrl=alg_ctrl)
    ops.append(op)

    # 3
    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'polynomial', 'distribution_index':0.20, 'probability': 0.1}  # Check
    crossover = {'name':'sbx', 'distribution_index':20, 'probability':1.0}  # Check
    selection = {'name':'bts'}
    termination = {'name':'n_eval','max_evaluations':max_evaluations}
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size':pop_size, 'selection':selection, 'mutation': mutation,'crossover': crossover, 'termination':termination}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_3', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    # 4
    op = Osyczka2_OptimizationProblem('OSY')
    mutation={'name':'polynomial', 'distribution_index':0.80, 'probability': 0.1}  # Check
    crossover = {'name':'sbx', 'distribution_index':20, 'probability':1.0}  # Check
    selection = {'name':'bts'}
    termination = {'name':'n_eval','max_evaluations':max_evaluations}
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size':pop_size, 'selection':selection, 'mutation': mutation,'crossover': crossover, 'termination':termination}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_4', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    # job
    job = MultibjectiveOptimizationComparer(name,max_workers,out_folder_path,None)
    job.execute()

def opttest_scipy(name,max_workers,out_folder_path):
    ops:List[OptimizationProblem] = []
    op = ('EX_16_4_COBYLA')
    opt_ctrl = {
        'method': 'COBYLA'}  # ovo je dictionary koji se šalje konstruktoru ScipyOptimizationAlgorithm-a.. to znaci da su ostale postavke defaultne..
    # opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)
    ops.append(op)
    op = EX_16_4_OptimizationProblem('EX_16_4_SLSQP')
    opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)
    ops.append(op)
    job = MultibjectiveOptimizationComparer(name, max_workers,out_folder_path, ops,None)
    job.execute()

def opttest_one_osy(name,max_workers,out_folder_path):
    ops: List[OptimizationProblem] = []
    pop_size = 200
    num_iter = 100
    max_evaluations = pop_size * num_iter
    #1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    job = MultibjectiveOptimizationComparer(name,max_workers,out_folder_path,None)
    job.execute()

def opttest_ref_front_osy(name,max_workers,out_folder_path):
    ops: List[OptimizationProblem] = []
    pop_size = 200
    num_iter = 600
    max_evaluations = pop_size * num_iter
    #1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_200_sol_ref_front', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    # Create and execute job
    job = MultibjectiveOptimizationComparer(name, max_workers,out_folder_path, ops,None)
    job.execute()

def opttest_one_osy_load(name,max_workers,out_folder_path,ref_out_path):
    ops: List[OptimizationProblem] = []
    pop_size = 200
    num_iter = 200
    max_evaluations = pop_size * num_iter
    #1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_one_osy', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    # load referent front
    ref_pareto_front = load_pareto_front(op, ref_out_path)
    job = MultibjectiveOptimizationComparerFromWrittenResults(name,max_workers,out_folder_path, ops, ref_pareto_front)
    job.execute()



if __name__ == '__main__':
    out_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
    max_number_of_workers = 4
    refoutfile='OSY_pymoo_nsga_ii_200_sol_ref_front.csv'
    ref_out_file_path = os.path.join(out_folder_path, refoutfile)
    #opttest_ref_front('test',max_number_of_workers,out_folder_path)
    #opttest_scipy('test',max_number_of_workers,out_folder_path)
    #opttest_osy('test',max_number_of_workers,out_folder_path)
    #opttest_one_osy('test',max_number_of_workers,out_folder_path)
    #opttest_ref_front_osy('test', max_number_of_workers, out_folder_path)
    opttest_one_osy_load('test', max_number_of_workers, out_folder_path, ref_out_file_path)
