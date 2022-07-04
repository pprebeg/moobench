import numpy as np
from jmetal.core.problem import FloatProblem
from optbase import OptimizationProblem,OptimizationAlgorithm,DesignVariable,DesignCriteria,DesignConstraint,OptimizationProblemSolution,DesignObjective
from typing import List, Dict
from abc import ABC, abstractmethod
from copy import copy, deepcopy

#IMPORT ALGORITHMS
from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.hype import HYPE
from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.algorithm.multiobjective.moead import  MOEAD
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.algorithm.multiobjective.spea2 import SPEA2

#IMPORT OPERATORS
from jmetal.operator.mutation import NullMutation, BitFlipMutation, PolynomialMutation, IntegerPolynomialMutation, SimpleRandomMutation, UniformMutation, NonUniformMutation, PermutationSwapMutation, CompositeMutation, ScrambleMutation
from jmetal.operator.crossover import NullCrossover, PMXCrossover, CXCrossover, SBXCrossover, IntegerSBXCrossover, SPXCrossover, DifferentialEvolutionCrossover, CompositeCrossover
from jmetal.operator.selection import RouletteWheelSelection, BinaryTournamentSelection, BestSolutionSelection, NaryRandomSolutionSelection, DifferentialEvolutionSelection, RandomSolutionSelection, RankingAndCrowdingDistanceSelection, RankingAndFitnessSelection, BinaryTournament2Selection

#IMPORT COMPARATORS
from jmetal.util.comparator import EqualSolutionsComparator, SolutionAttributeComparator, MultiComparator, RankingAndCrowdingDistanceComparator, StrengthAndKNNDistanceComparator, OverallConstraintViolationComparator, DominanceComparator, GDominanceComparator, EpsilonDominanceComparator

#IMPORT TERMINATION CRITERIA
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime, StoppingByKeyboard, StoppingByQualityIndicator



class jmetalConstraint():

    '''This class normalizes the constraints embody'''
    
    def __init__(self, con:DesignConstraint):    #tu bi se trebalo kod constrainta koristiti ono kaj je moja ideja bila na seminarskom.. con mora biti funkcija
        self._criteria:DesignConstraint=con

    @property
    def con(self)->DesignConstraint:
        return self._criteria

    def get_con_value(self) ->float:
        return self.con.value_gt_0         #slicno kao u scipy-ju! expression treba biti >=0, ako je <0, smatra se violation-om - to se vidi u jmetal.util.constraint_handling

    def get_con_value_normalized(self) ->float:
        return self.con.value_gt_0/self.con.rhs         #slicno kao u scipy-ju! expression treba biti >=0, ako je <0, smatra se violation-om - to se vidi u jmetal.util.constraint_handling/self.con.rhs 

class WrappedjmetalProblem(FloatProblem):

    '''Class that inherits from Problem in order to instantiate an object that will define necessary methods for jmetal algorithm calculations. In the constructor - using super() function, 5 obligatory
    arguments are passed. Those are: how many design variables are there, how many objectives and constraints, as well as arrays of lower and upper boundaries. On top of that, two callback functions are passed. Those are
    called in _evaluate method. Crucial method that starts the calculation of problem.'''

    def __init__(self, desvars:List[DesignVariable], objs:List[DesignObjective], cons:List[jmetalConstraint], xl:np.ndarray, xu:np.ndarray, callback_evaluate, callback_get_current_solution): #sve DesignVariable, DesignConstraint i ostalo pohranjeno je u OptimizationProblem klasi. Ako ce trebati bas jmetal wrapperi tih klasa, onda se ovdje budu dodijelile.

        self._callback_evaluate=callback_evaluate
        self._callback_get_current_solution=callback_get_current_solution

        #DEFINIRANJE POTREBNIH ATRIBUTA klase Problem

        self.number_of_variables = len(desvars)
        self.number_of_objectives = len(objs)
        self.number_of_constraints = len(cons)
        self.lower_bound = list(xl)
        self.upper_bound = list(xu)
        
##        FloatSolution.lower_bound = self.lower_bound
##        FloatSolution.upper_bound = self.upper_bound

        #SPREMANJE LISTA OBJEKATA tipa DesignVariable, DesignObjective, jmetalConstraint

        self._desvars = desvars
        self._cons:jmetalConstraint = cons
        self._objs = objs




    def evaluate(self, solution):

        x = solution.variables

        self._callback_evaluate(x)

        curr_solution=self._callback_get_current_solution() # u ovaj solution se naime spremaju u

        objectives=[]            #faster way to initialize a list arr = [0]*1000 ako postoji nacin overwrite vrijednosti?
        constraints=[]              # je li dobro ovo ovako prebrisivat svaki put?

        for i in range(self.number_of_objectives):
            objectives.append(curr_solution.get_obj_value(i))

        for i in range(self.number_of_constraints):
            constraints.append(self._cons[i].get_con_value_normalized())    #Zasad je implementirano samo ovo normalizirano vracanje vrijednosti! Sve ostalo bi dosta usporavalo! 

        solution.objectives = objectives

        solution.constraints = constraints #constraints trebaju biti oblika >=0

        return solution

    def get_name(self) -> str:
        return 'WrappedjmetalpyProblem'



class jmetalOptimizationAlgorithm(OptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a jmetal problem that is suitable for jmetal interface - function minimize. '''

    def __init__(self, name:str, method:str, alg_ctrl:Dict=None):

        super().__init__(name)

        self._method=method.lower()


        mutation = None
        crossover = None
        sampling = None
        selection = None
        termination = None

        #Dictionary s postavkama

        self._alg_options = {}

        self._sol = None
        
        
        selection = alg_ctrl.get('selection')
        crossover = alg_ctrl.get('crossover')
        mutation = alg_ctrl.get('mutation')
        termination = alg_ctrl.get('termination')
        sampling = alg_ctrl.get('sampling')
        
        if sampling != None:
            alg_ctrl.pop('sampling')
            sampling_obj = self._generate_sampling(sampling)
            self._alg_options['sampling'] = sampling_obj


        if selection != None:
            alg_ctrl.pop('selection')
            selection_obj = self._generate_selection(selection)
            self._alg_options['selection'] = selection_obj


        if crossover != None:
            alg_ctrl.pop('crossover')
            print(crossover)
            crossover_obj = self._generate_crossover(crossover)
            self._alg_options['crossover'] = crossover_obj

        if mutation != None:
            alg_ctrl.pop('mutation')
            mutation_obj = self._generate_mutation(mutation)
            self._alg_options['mutation'] = mutation_obj            
          
        if termination != None:
            alg_ctrl.pop('termination')
            self._termination = self._generate_termination(termination)
            self._alg_options['termination_criterion'] = self._termination           
            


        self._alg_options.update(alg_ctrl)
             


    @property
    def sol(self):

        return self._sol        # ovo ce biti ono sto vraca jmetal funkcija minimize.. ovdje - jmetal.Result

    def _generate_jmetal_problem(self,
                desvars: List[DesignVariable],
                constraints: List[DesignConstraint],
                objectives: List[DesignObjective],
                callback_evaluate,
                callback_get_current_solution)-> WrappedjmetalProblem:

        '''This function is crucial to transform general problem, algorithm definitions, variables, etc. defined through optbase.py. into jmetal specific problem, algorithm and so on. This, and only this type of data
            can than be passed through to jmetal. '''


        # Design Variables and Bounds
        xl: List[float] = []
        xu: List[float] = []

        xl, xu=self._get_bounds(desvars)    #kreiranje niza za donje i gornje granice varijabli

        #Constraints
        jmetal_cons:List[jmetalConstraint]=[]
        for con in constraints:
            jmetalcon = jmetalConstraint(con)
            jmetal_cons.append(jmetalcon)

        problem = WrappedjmetalProblem(desvars, objectives, jmetal_cons, xl, xu, callback_evaluate, callback_get_current_solution) #ovaj callback_evaluate

        return problem

    def _instantiate_jmetal_algorithm_object(self, problem, jmetal_algorithm_options):

        if self._method == 'gde3':
            return GDE3(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'hype':
            return HYPE(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'ibea':
            return IBEA(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'mocell':
            return MOCell(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'moea/d':
            return MOEAD(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'nsga2':
            return NSGAII(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'nsga3':
            return NSGAIII(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'spea2':

            return SPEA2(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'omopso':
            returnOMOPSO(**jmetal_algorithm_options)
            return OMOPSO(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'smpso':
            return SMPSO(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'es':
            return EvolutionStrategy(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'ga':
            
            return GeneticAlgorithm(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'ls':
            return LocalSearch(problem=problem, **jmetal_algorithm_options)

        elif self._method == 'sa':
            return SimulatedAnnealing(problem=problem, **jmetal_algorithm_options)

    def _generate_crossover(self, item):

        type_of_crossover:str = item.get('name')
        item.pop('name')
        
        if type_of_crossover == 'null':
            crossover = NullCrossover(**item)
        elif type_of_crossover == 'pmx':    #probability
            crossover = PMXCrossover(**item)
        elif type_of_crossover == 'cx':     #probability
            crossover = CXCrossover(**item)
        elif type_of_crossover == 'sbx':    #probability, distribution index (oblik 20.0)
            crossover = SBXCrossover(**item)
        elif type_of_crossover == 'int_sbx':    #probability, distribution index (oblik 20.0)
            crossover = IntegerSBXCrossover(**item)
        elif type_of_crossover == 'spx':    #probability
            crossover = SPXCrossover(**item)
        elif type_of_crossover == 'de':     #CR, F i K
            crossover = DifferentialEvolutionCrossover(**item)
        elif type_of_crossover == 'composite':  #crossover_operator_list
            crossover = CompositeCrossover(**item)

        return crossover

    def _generate_mutation(self, item:Dict):

        type_of_mutation:str = item.get('name')
        item.pop('name')
        
        if type_of_mutation == 'null':
            mutation = NullMutation(**item)
        elif type_of_mutation == 'bitflip': #probability
            mutation = BitFlipMutation(**item)
        elif type_of_mutation == 'polynomial':  #probability, distribution_index: float = 0.20
            mutation = PolynomialMutation(**item)
        elif type_of_mutation == 'integer_polynomial':  #probability, distribution_index: float = 0.20
            mutation = IntegerPolynomialMutation(**item)
        elif type_of_mutation == 'simple_random':   #probability
            mutation = SimpleRandomMutation(**item)
        elif type_of_mutation == 'uniform': # probability, perturbation 0.5
            mutation = UniformMutation(**item)
        elif type_of_mutation == 'non_uniform': # probability, perturbation: float = 0.5, max_iterations: int = 0.5
            mutation = NonUniformMutation(**item)
        elif type_of_mutation == 'permutation_swap':    # ?? nema konstruktor ??
            mutation = PermutationSwapMutation(**item)
        elif type_of_mutation == 'composite':   # mutation_operator_list:[Mutation]
            mutation = CompositeMutation(**item)
        elif type_of_mutation == 'scramble':    # solution: PermutationSolution
            mutation = ScrambleMutation(**item)

        return mutation

    def _generate_selection(self, item):
        
        type_of_selection:str = item.get('name')
        item.pop('name')

        #poziv kreiranja comparatora! 
##        type_of_comparator = item.get('comparator')
##        item.pop('comparator')
##        if type_of_comparator == 'esc':
##            comparator = EqualSolutionsComparator()
##            item.['comparator'] = comparator
##            pass
##        elif type_of_comparator == 'sac':
##            comparator = SolutionAttributeComparator()
##            item.['comparator'] = comparator
##            pass
        
        if type_of_selection == 'rws':
            selection = RouletteWheelSelection(**item)
        elif type_of_selection == 'bts':
            selection = BinaryTournamentSelection(**item)
        elif type_of_selection == 'bss':  #comparator: Comparator = DominanceComparator()
            selection = BestSolutionSelection(**item)
        elif type_of_selection == 'nrss':  #number_of_solutions_to_be_returned
            selection = NaryRandomSolutionSelection(**item)
        elif type_of_selection == 'des':   #
            selection = DifferentialEvolutionSelection(**item)
        elif type_of_selection == 'rss': 
            selection = RandomSolutionSelection(**item)
        elif type_of_selection == 'rcds': #max_population_size: int, dominance_comparator: Comparator = DominanceComparator()
            selection = RankingAndCrowdingDistanceSelection(**item)
        elif type_of_selection == 'rfs':    #                  max_population_size: int, reference_point: S, dominance_comparator: Comparator = DominanceComparator()
            selection = RankingAndFitnessSelection(**item)
        elif type_of_selection == 'bt2s':   # comparator_list: List[Comparator]
            selection = BinaryTournament2Selection(**item)

        return selection

    def _generate_termination(self, item):

        type_of_termination:str = item.get('name')
        item.pop('name')

        #generiranje quality indicatora! 
        
        if type_of_termination == 'n_eval':
            termination = StoppingByEvaluations(**item)
        elif type_of_termination == 'time': #max_seconds
            termination = StoppingByTime(**item)
        elif type_of_termination == 'keyboard':  #
            termination = StoppingByKeyboard(**item)
        elif type_of_termination == 'qi':  #quality_indicator: QualityIndicator, expected_value: float, degree: float
            termination = StoppingByQualityIndicator(**item)
            
        return termination


    def _get_bounds(self, dvs:List[DesignVariable]):

        lbs = []
        ubs = []
        for dv in dvs:
            lb,ub = (dv.get_bounds())
            lbs.append(lb)
            ubs.append(ub)

        return np.array(lbs), np.array(ubs)

    def _get_current_desvar_values(self, dvs:List[DesignVariable]):
        x0 = []
        for jmetal_dv in jmetal_dvs:
            x0.append(jmetal_dv.value)
        return x0
    
    @abstractmethod
    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        pass

class jmetalOptimizationAlgorithmMulti(jmetalOptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a jmetal problem that is suitable for jmetal interface - function minimize. '''

    def __init__(self,name:str, method:str, alg_ctrl:Dict=None):

        super().__init__(name, method, alg_ctrl)
        

    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        problem = None

        problem = self._generate_jmetal_problem(desvars, constraints, objectives, callback_evaluate, callback_get_current_solution)  #Definiranje jmetal problem

        if problem==None:
            print('Problem is not defined!')
            raise

        self._algorithm = self._instantiate_jmetal_algorithm_object(problem, self._alg_options)

        self._algorithm.run()

        sol = self._algorithm.get_result()
        self._sol = sol

        #FINAL EVALUATION OF OPTIMAL SOLUTION TO BE STORED AS OptimizationProblemSolution

        solutions:List[OptimizationProblemSolution] = []
        
        for solution_ in sol:
            
            problem.evaluate(solution_)     #ovo bi trebalo spremiti u OptimizationProblemSolution, OptimizationProblem
            opt_sol:OptimizationProblemSolution = callback_get_current_solution() #vraca OptimizationProblemSolution koji je optbase objekt za cuvanje rjesenja u numerickom obliku. ili izraditi copy objekta - funckionalnost na razini pymoo_proto.. ili u optbase prosiriti poziv ovog optimize-a sa jos jednim callbackom koji bi pozivao add_
            solutions.append(deepcopy(opt_sol))   #append adds a reference only! in solutions, there are just pointers to opt_sol! it should actually make copies that are independent one of another, so that it doesn't change when opt_sol change                
    
        return solutions  

class jmetalOptimizationAlgorithmSingle(jmetalOptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a jmetal problem that is suitable for jmetal interface - function minimize. '''

    def __init__(self,name:str, method:str, alg_ctrl:Dict=None):

        super().__init__(name, method, alg_ctrl)

    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        problem = None

        problem = self._generate_jmetal_problem(desvars, constraints, objectives, callback_evaluate, callback_get_current_solution)  #Definiranje jmetal problem

        if problem==None:
            print('Problem is not defined!')
            raise
        

        self._algorithm = self._instantiate_jmetal_algorithm_object(problem, self._alg_options)


        self._algorithm.run()

        sol = self._algorithm.get_result()
        
        self._sol = sol

        problem.evaluate(sol)

        opt_sol:OptimizationProblemSolution = callback_get_current_solution() #vraca objekt OptimizationProblemSolution koji je optbase objekt za cuvanje rjesenja u cistom numerickom obliku.

        solutions:List[OptimizationProblemSolution] = []

        solutions.append(opt_sol)

        return solutions
