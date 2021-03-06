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
    op = Analiza_okvira_OptimizationProblem('Frame')
    pop_size = 500
    mutation = {'name':'real_pm', 'eta':20, 'prob': 1/op.num_var+0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    
    termination = ('time', '02:00:00')
    
    alg_ctrl = {'pop_size': pop_size,'n_offsprings':10, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1','nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    
    #JOB
    job = MultibjectiveOptimizationComparer(name,max_workers,out_folder_path,None)
    job.execute()

def opttest_frame_problem(name:str, max_workers:int, out_folder_path:str,
                          ops:List[OptimizationProblem],
                          ref_pareto_front:OptimizationProblemMultipleSolutions):
    
    ops:List[OptimizationProblem] = []
    pop_size = 100
    num_iter = 400
    max_evaluations = pop_size * num_iter

    #PYMOO

    #1
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.6}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1','nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    #2
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.6}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_2','nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

##    #3
##    op = Analiza_okvira_OptimizationProblem('Frame')
##    
##    mutation = {'name':'real_pm', 'eta':20, 'prob': 1/op.num_var*10}  # Check
##    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
##    termination = ('n_eval', max_evaluations)
###    ref_points = np.array([[0.5, 0.2], [0.1, 0.6]]) #jasno da treba masu drugacije normalizirati tako da mo??emo provrtiti
##    
##    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
##    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_r-nsga_ii_1','rnsga2', alg_ctrl=alg_ctrl)
##    ops.append(op)

    #4
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.6}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    ref_dirs = {'name':'das-dennis', 'n_dim':op.num_obj, 'n_partitions':24} #moze se poslati i numpy array u dictionary alg_ctrl kao 'ref_dirs':numpy.array
    
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination, 'ref_dirs':ref_dirs}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_u-nsga_iii_1','unsga3', alg_ctrl=alg_ctrl)
    ops.append(op)

    #5
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.6}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    ref_dirs = {'name':'das-dennis', 'n_dim':op.num_obj, 'n_partitions':24} #moze se poslati i numpy array u dictionary alg_ctrl kao 'ref_dirs':numpy.array
    
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation,'crossover': crossover, 'termination':termination, 'ref_dirs':ref_dirs}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_r-nsga_iii_1','unsga3', alg_ctrl=alg_ctrl)
    ops.append(op)

    #6 - ZASAD CTAE ne izvr??ava uspje??no
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.6}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    ref_dirs = {'name':'das-dennis', 'n_dim':op.num_obj, 'n_partitions':24} #moze se poslati i numpy array u dictionary alg_ctrl kao 'ref_dirs':numpy.array
    
    alg_ctrl = { 'mutation': mutation,'crossover': crossover, 'termination':termination, 'ref_dirs':ref_dirs}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_c-taea_1','ctaea', alg_ctrl=alg_ctrl)
    ops.append(op)

    #JMETALPY

    #1
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    termination = {'name':'n_eval', 'max_evaluations':max_evaluations}
    
    alg_ctrl = {'population_size': pop_size, 'cr': 0.5, 'f': 0.5, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_gde3_1','gde3', alg_ctrl=alg_ctrl)
    ops.append(op)

##    #2
##    op = Analiza_okvira_OptimizationProblem('Frame')
##
##    mutation = {'name':'polynomial','probability':1.0 / op.num_var, 'distribution_index':20}
##    crossover = {'name':'sbx','probability':1.0, 'distribution_index':20}
##    termination = {'name':'n_eval', 'max_evaluations':max_evaluations}
##    
####    reference_point = {'name':'float', 'lower_bound': [,], 'upper_bound': [,], 'number_of_objectives': op.num_obj, 'number_of_constraints': op.num_con}
##    alg_ctrl = {'population_size': pop_size,'offspring_population_size':pop_size, 'mutation': mutation, 'crossover': crossover, 'termination_criterion':termination, 'reference_point':reference_point}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
##    op.opt_algorithm = PymooOptimizationAlgorithmMulti('jmetalpy_hype_1','hype', alg_ctrl=alg_ctrl)
##    ops.append(op)

    #3
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    mutation = {'name':'polynomial','probability':0.6, 'distribution_index':0.20}
    crossover = {'name':'sbx','probability':1.0, 'distribution_index':20}
    termination = {'name':'n_eval', 'max_evaluations':max_evaluations}
    
    alg_ctrl = {'population_size': pop_size,'offspring_population_size':pop_size, 'kappa':1, 'mutation': mutation, 'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_ibea_1','ibea', alg_ctrl=alg_ctrl)
    ops.append(op)


##    #4
##    op = Analiza_okvira_OptimizationProblem('Frame')
##
##    mutation = {'name':'polynomial','probability':1.0 / op.num_var, 'distribution_index':20}
##    crossover = {'name':'sbx','probability':1.0, 'distribution_index':20}
##    termination = {'name':'n_eval', 'max_evaluations':max_evaluations}
##    
##    alg_ctrl = {'population_size': pop_size,'offspring_population_size':pop_size, 'mutation': mutation, 'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
##    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_spea2_1','spea2', alg_ctrl=alg_ctrl)
##    ops.append(op)

    #5
    op = Analiza_okvira_OptimizationProblem('Frame')
    
    neighborhood={'name':'c9','rows':10,'columns':10}
    archive={'name':'crowding_distance', 'maximum_size':100}
    mutation = {'name':'polynomial','probability':0.6, 'distribution_index':0.2}
    crossover = {'name':'sbx','probability':1.0, 'distribution_index':20}
    termination = {'name':'n_eval', 'max_evaluations':max_evaluations}

    alg_ctrl = {'population_size': pop_size, 'neighborhood':neighborhood, 'archive':archive, 'mutation': mutation, 'crossover': crossover, 'termination':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_mocell_1','mocell', alg_ctrl=alg_ctrl)
    ops.append(op)


    #6
##algorithm = MOEAD(
##    problem=problem,
##    population_size=300,
##    crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
##    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
##    aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
##    neighbor_size=20,
##    neighbourhood_selection_probability=0.9,
##    max_number_of_replaced_solutions=2,
##    weight_files_path='resources/MOEAD_weights',
##    termination_criterion=StoppingByEvaluations(max=max_evaluations)
##)
##    op = Analiza_okvira_OptimizationProblem('Frame')
##    
##    neighborhood={'name':'c9','rows':10,'columns':10}
##    archive={'name':'crowding_distance', 'maximum_size':100}
##    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
##    crossover=SBXCrossover(probability=1.0, distribution_index=20),
##    termination_criterion=StoppingByEvaluations(max=max_evaluations)
##
##    op = Analiza_okvira_OptimizationProblem('Frame')
##
##    mutation = {'name':'polynomial','probability':1.0 / op.num_var, 'distribution_index':20}
##    crossover = {'name':'de','CR':1.0, 'F':0.5, 'K':0.5}
##    termination = {'name':'n_eval', 'max_evaluations':max_evaluations)
##    
##    alg_ctrl = {'population_size': pop_size,'offspring_population_size':pop_size, 'neighborhood':neighborhood, 'archive':archive, 'mutation': mutation, 'crossover': crossover, 'termination_criterion':termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
##    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_mocell_1','mocell', alg_ctrl=alg_ctrl)
##    ops.append(op)

    
    # JOB - load reference front, create job, execute job
    ref_pareto_front = load_pareto_front(op, ref_out_path)
    job = MultibjectiveOptimizationComparerFromWrittenResults(name, max_workers, out_folder_path, ops, ref_pareto_front)
    job.execute()


def opttest_osy(name,max_workers,out_folder_path):
    ops: List[OptimizationProblem] = []
    pop_size = 100
    num_iter = 10
    max_evaluations = pop_size * num_iter

    #1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    #2
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_2', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    #3
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_3', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    #4
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name': 'real_pm', 'eta': 20, 'prob': 0.1}  # Check
    crossover = {'name': 'real_sbx', 'eta': 20, 'prob': 0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_4', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    # 5
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_5', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    # 6
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_6', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    # 7
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_7', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    # 8
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_8', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    # job
    job = MultibjectiveOptimizationComparer(name,max_workers,out_folder_path,ops,None)
    ops2 = job.execute()

    return ops2

def opttest_scipy(name,max_workers,out_folder_path):
    ops:List[OptimizationProblem] = []
    op = ('EX_16_4_COBYLA')
    opt_ctrl = {
        'method': 'COBYLA'}  # ovo je dictionary koji se ??alje konstruktoru ScipyOptimizationAlgorithm-a.. to znaci da su ostale postavke defaultne..
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
    max_evaluations = pop_size * num_iter
    
    #2
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}  # Check
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)

    #JOB
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
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.5}
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.8}
    termination = ('n_eval', max_evaluations)
    alg_ctrl = {'pop_size': pop_size, 'n_offsprings': 10, 'mutation': mutation, 'crossover': crossover,
                'termination': termination}  
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_one_osy', 'nsga2', alg_ctrl=alg_ctrl)
    ops.append(op)
    
    # load referent front
    ref_pareto_front = load_pareto_front(op, ref_out_path)
    job = MultibjectiveOptimizationComparerFromWrittenResults(name,max_workers,out_folder_path, ops, ref_pareto_front)
    job.execute()



if __name__ == '__main__':
    out_folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
    max_number_of_workers = 5
    refoutfile='OSY_pymoo_nsga_ii_200_sol_ref_front.csv'
    ref_out_file_path = os.path.join(out_folder_path, refoutfile)
    #opttest_ref_front('test',max_number_of_workers,out_folder_path)
    #opttest_scipy('test',max_number_of_workers,out_folder_path)
    #opttest_osy('test',max_number_of_workers,out_folder_path)
    #opttest_one_osy('test',max_number_of_workers,out_folder_path)
##    opttest_ref_front_osy('test', max_number_of_workers, out_folder_path)
##    opttest_one_osy_load('test', max_number_of_workers, out_folder_path, ref_out_file_path)
    #opttest_ref_front(max_number_of_workers)
    #opttest_scipy(max_number_of_workers)
    ops2 = opttest_osy('testing',max_number_of_workers, out_folder_path)
    #opttest_one_osy(max_number_of_workers)
    #opttest_frame_problem('Final test', max_number_of_workers, out_folder_path, ref_out_file_path)
