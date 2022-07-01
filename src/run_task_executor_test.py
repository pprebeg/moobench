from ex_16_4_optimization_problem import *
from osyczka2 import Osyczka2_OptimizationProblem
from concurrent.futures import ProcessPoolExecutor,Future
from typing import List, Dict
from jobbase import MultibjectiveOptimizationComparer
from pymoo.factory  import get_mutation, get_crossover, get_selection
from optlib_pymoo_proto import PymooOptimizationAlgorithmMulti
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByTime, StoppingByEvaluations, StoppingByQualityIndicator,StoppingByKeyboard
from jmetal.core.quality_indicator import FitnessValue, GenerationalDistance, InvertedGenerationalDistance, EpsilonIndicator, HyperVolume
from optlib_jmetalpy_proto import jmetalOptimizationAlgorithmMulti

def opttest1():
    ops:List[OptimizationProblem] = []
    op = Osyczka2_OptimizationProblem('OSY')
    pop_size = 100
    num_iter = 10
    max_evaluations = pop_size * num_iter
    mutation_obj = get_mutation('real_pm', eta=20, prob=1.0 / 6)  # Check
    crossover_obj = get_crossover('real_sbx', eta=20, prob=1.0)  # Check
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation_obj,
                'crossover': crossover_obj}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl = {'n_eval': max_evaluations}  # Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_1','nsga2', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)
    ops.append(op)
    #2
    op = Osyczka2_OptimizationProblem('OSY')
    mutation_obj = get_mutation('real_pm', eta=20, prob=1.0 / 4)  # Check
    crossover_obj = get_crossover('real_sbx', eta=20, prob=0.9)  # Check
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation_obj,
                'crossover': crossover_obj}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl = {'n_eval': max_evaluations}  # Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_2','nsga2', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)
    ops.append(op)
    #3
    op = Osyczka2_OptimizationProblem('OSY')
    mutation_obj = get_mutation('real_pm', eta=20, prob=1.0 / 10)  # Check
    crossover_obj = get_crossover('real_sbx', eta=25, prob=0.95)  # Check
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation_obj,
                'crossover': crossover_obj}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl = {'n_eval': max_evaluations}  # Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_3','nsga2', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)
    ops.append(op)
    #4
    op = Osyczka2_OptimizationProblem('OSY')
    mutation_obj = get_mutation('real_pm', eta=25, prob=1.0 / 6)  # Check
    crossover_obj = get_crossover('real_sbx', eta=25, prob=1.0)  # Check
    alg_ctrl = {'pop_size': pop_size, 'mutation': mutation_obj,
                'crossover': crossover_obj}  # u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl = {'n_eval': max_evaluations}  # Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('pymoo_nsga_ii_4','nsga2', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)
    ops.append(op)
    # 1
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = PolynomialMutation(probability=1.0 / 6, distribution_index=0.20)
    crossover = SBXCrossover(probability=1.0, distribution_index=0.20)
    selection = BinaryTournamentSelection()
    operators = {'mutation': mutation, 'crossover': crossover, 'selection': selection}
    termination_criterion = StoppingByEvaluations(max_evaluations)
    alg_ctrl = {'population_size': pop_size,'offspring_population_size': pop_size}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_1','nsga-ii', operators=operators, alg_ctrl=alg_ctrl, termination_criterion=termination_criterion)
    ops.append(op)
    # 2
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = PolynomialMutation(probability=1.0 / 8, distribution_index=0.20)
    crossover = SBXCrossover(probability=0.9, distribution_index=0.20)
    selection = BinaryTournamentSelection()
    operators = {'mutation': mutation, 'crossover': crossover, 'selection': selection}
    termination_criterion = StoppingByEvaluations(max_evaluations)
    alg_ctrl = {'population_size': pop_size,'offspring_population_size': pop_size}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_2','nsga-ii', operators=operators, alg_ctrl=alg_ctrl,termination_criterion=termination_criterion)
    ops.append(op)
    # 3
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = PolynomialMutation(probability=1.0 / 10, distribution_index=0.20)
    crossover = SBXCrossover(probability=0.95, distribution_index=0.25)
    selection = BinaryTournamentSelection()
    operators = {'mutation': mutation, 'crossover': crossover, 'selection': selection}
    termination_criterion = StoppingByEvaluations(max_evaluations)
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size': pop_size}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_3','nsga-ii', operators=operators, alg_ctrl=alg_ctrl,
                                                        termination_criterion=termination_criterion)
    ops.append(op)
    # 4
    op = Osyczka2_OptimizationProblem('OSY')
    mutation = PolynomialMutation(probability=1.0 / 20, distribution_index=0.3)
    crossover = SBXCrossover(probability=0.95, distribution_index=0.22)
    selection = BinaryTournamentSelection()
    operators = {'mutation': mutation, 'crossover': crossover, 'selection': selection}
    termination_criterion = StoppingByEvaluations(max_evaluations)
    alg_ctrl = {'population_size': pop_size, 'offspring_population_size': pop_size}
    op.opt_algorithm = jmetalOptimizationAlgorithmMulti('jmetalpy_nsga_ii_4','nsga-ii', operators=operators, alg_ctrl=alg_ctrl,
                                                        termination_criterion=termination_criterion)
    ops.append(op)



    #job
    max_number_of_workers = 4
    job = MultibjectiveOptimizationComparer('test',ops,max_number_of_workers)
    job.execute()

def opttest2():
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
    job = MultibjectiveOptimizationComparer('test',ops,4)
    job.execute()

if __name__ == '__main__':
    opttest1()