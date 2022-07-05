from ex_16_4_optimization_problem import *
from Frame_problem import Analiza_okvira_OptimizationProblem
from osyczka2 import Osyczka2_OptimizationProblem
from concurrent.futures import ProcessPoolExecutor,Future
from typing import List, Dict
from jobbase import MultibjectiveOptimizationComparer
from optlib_pymoo_proto import PymooOptimizationAlgorithmMulti
from optlib_jmetalpy_proto import jmetalOptimizationAlgorithmMulti
from optlib_scipy import ScipyOptimizationAlgorithm

def opttest_ref_front(max_number_of_workers:int=2):
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
    job = MultibjectiveOptimizationComparer('test',ops,max_number_of_workers)
    job.execute()


def opttest_osy(max_number_of_workers:int=2):
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
    job = MultibjectiveOptimizationComparer('test', ops, max_number_of_workers)
    job.execute()

def opttest_scipy(max_number_of_workers:int=2):
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
    job = MultibjectiveOptimizationComparer('test',ops,max_number_of_workers)
    job.execute()

if __name__ == '__main__':
    max_number_of_workers = 4
    #opttest_ref_front(max_number_of_workers)
    #opttest_scipy(max_number_of_workers)
    opttest_osy(max_number_of_workers)
