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

#IMPORT TERMINATION CRITERIA
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime, StoppingByKeyboard, StoppingByQualityIndicator



class jmetalConstraint():
    def __init__(self, con:DesignConstraint):    #tu bi se trebalo kod constrainta koristiti ono kaj je moja ideja bila na seminarskom.. con mora biti funkcija
        self._criteria:DesignConstraint=con

    @property
    def con(self)->DesignConstraint:
        return self._criteria

    def get_con_value(self) ->float:
        return self.con.value_gt_0          #slicno kao u scipy-ju! expression treba biti >=0, ako je <0, smatra se violation-om.

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
        self._cons = cons
        self._objs = objs




    def evaluate(self, solution):

        x = solution.variables

        self._callback_evaluate(x)

        curr_solution=self._callback_get_current_solution() # u ovaj solution se naime spremaju u

        objectives=[]            #faster way to initialize a list arr = [0]*1000 ako postoji nacin overwrite vrijednosti?
        constraints=[] # je li dobro ovo ovako prebrisivat svaki put?

        for i in range(self.number_of_objectives):
            objectives.append(curr_solution.get_obj_value(i))

        for i in range(self.number_of_constraints):
            constraints.append(self._cons[i].get_con_value())

        solution.objectives = objectives

        solution.constraints = constraints #izgleda da constraints trebaju biti oblika >=0

        return solution

    def get_name(self) -> str:
        return 'WrappedjmetalpyProblem'



class jmetalOptimizationAlgorithm(OptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a jmetal problem that is suitable for jmetal interface - function minimize. '''

    def __init__(self, method:str, operators:Dict=None, alg_ctrl:Dict=None, termination_criterion=None):

        self._method=method.lower()
        self._selection = None
        self._crossover = None
        self._mutation = None

        self._termination_criterion = termination_criterion

        #Dictionary s postavkama
        self._alg_options = alg_ctrl
        self._sol = None
        
        
        self._selection = operators.get('selection')
        self._crossover = operators.get('crossover')
        self._mutation = operators.get('mutation')

         #dictionary self._options se raspakirava u keyword arguments. Nema greske ako je prazan dictionary..
                                                                            #self._opt_ctrl_iterables - ovo mozda nije prikladno! Tj. treba ipak prilagodba za algoritme pojedine.
                                                                            #Npr. RNSGA2 algoritam treba ref_points- tj. treba np.ndarray na drugom mjestu. al to samo taj algoritam, pa eto tome treba prilagoditi

        #POSTAVKE (ATRIBUTI) za kreiranje TerminationCriterion! - ako moguce vise ih postaviti! onda elif zamijeniti sa if!
##        if self._termination!=None:
##            if self._termination.get('n_eval')!=None:
##                value=self._termination.get('n_eval')
##                self._termination=('n_eval',value)
##
##            elif self._termination.get('keyboard')!=None:
##                self._termination=('keyboard')
##
##            elif self._termination.get('time')!=None:
##                value=self._termination.get('time')
##                self._termination=('time',value)
##

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

    def _instantiate_jmetal_algorithm_object(self, jmetal_algorithm_options):

        if self._method == 'gde3':
            return GDE3(**jmetal_algorithm_options)

        elif self._method == 'hype':
            return HYPE(**jmetal_algorithm_options)

        elif self._method == 'ibea':
            return IBEA(**jmetal_algorithm_options)

        elif self._method == 'mocell':
            return MOCell(**jmetal_algorithm_options)

        elif self._method == 'moea/d':
            return MOEAD(**jmetal_algorithm_options)

        elif self._method == 'nsga-ii':
            return NSGAII(**jmetal_algorithm_options)

        elif self._method == 'nsga-iii':
            return NSGAIII(**jmetal_algorithm_options)

        elif self._method == 'spea2':
            return SPEA2(**jmetal_algorithm_options)

        elif self._method == 'omopso':
            return OMOPSO(**jmetal_algorithm_options)

        elif self._method == 'smpso':
            return SMPSO(**jmetal_algorithm_options)

        elif self._method == 'es':
            return EvolutionStrategy(**jmetal_algorithm_options)

        elif self._method == 'ga':
            
            return GeneticAlgorithm(**jmetal_algorithm_options)

        elif self._method == 'ls':
            return LocalSearch(**jmetal_algorithm_options)

        elif self._method == 'sa':
            return SimulatedAnnealing(**jmetal_algorithm_options)

    def _get_mutation(self):

        if self._mutation[0]==None:
            return NullMutation()
        elif self._mutation[0]=='':
            pass

    def _get_crossover(self):
        pass

    def _get_selection(self):                   #druga opcija definiranja 

        
        selection_type = self._selection[0].lower()

        #SETTINGS DICTIONARY  
        settings:Dict=self._selection[1]
        
        if selection_type=='rws':        #ili da ovo ipak user sam kreira pa pošalje ovoj biblioteci iz korisnickog programa?! pokusati tako  
            return RouletteWheelSelection(**settings)   #za settingse se onda koriste keyword argumenti kakve bi koristili i direktno.. tipa probability=0.2
        elif selection_type=='bts':
            return BinaryTournamentSelection(**settings)
        elif selection_type=='bss':
            return BestSolutionSelection(**settings)
        elif selection_type=='nrss':
            return NaryRandomSolutionSelection(**settings)
        elif selection_type=='des':
            return DifferentialEvolutionSelection(**settings)
        elif selection_type=='rss':
            return RandomSolutionSelection(**settings)
        elif selection_type=='rcds':
            return RankingAndCrowdingDistanceSelection(**settings)
        elif selection_type=='rfs':
            return RankingAndFitnessSelection(**settings)
        elif selection_type=='bt2s':
            return BinaryTournament2Selection(**settings)


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

    def __init__(self, method:str, operators:Dict=None, alg_ctrl:Dict=None, termination_criterion=None):

        super().__init__(method, operators, alg_ctrl, termination_criterion)
        

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

        #OPERATORS
        mutation = self._mutation
        crossover = self._crossover
        selection = self._selection

        #TERMINATION
        termination_criterion = self._termination_criterion
        
        jmetal_algorithm_options={}

        if problem != None:
            jmetal_algorithm_options['problem']=problem
        if mutation != None:
            jmetal_algorithm_options['mutation']=mutation
        if crossover != None:
            jmetal_algorithm_options['crossover']=crossover
        if selection != None:
            jmetal_algorithm_options['selection']=selection
        if termination_criterion != None:
            jmetal_algorithm_options['termination_criterion']=termination_criterion

        #self._create_termination_criterion()

        jmetal_algorithm_options.update(self._alg_options)
            
        self._algorithm = self._instantiate_jmetal_algorithm_object(jmetal_algorithm_options) #sto ako nema tog parametra mozda try catch, pa putem prepoznavanja koji argument ne postoji, ponovno pozvat kreiranje algoritma

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

    def __init__(self, method:str, operators:Dict=None, alg_ctrl:Dict=None, termination_criterion=None):

        super().__init__(method, operators, alg_ctrl, termination_criterion)

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

        #OPERATORS
        mutation = self._mutation
        crossover = self._crossover
        selection = self._selection

        #TERMINATION
        termination_criterion = self._termination_criterion
        
        jmetal_algorithm_options={}

        if problem != None:
            jmetal_algorithm_options['problem']=problem
        if mutation != None:
            jmetal_algorithm_options['mutation']=mutation
        if crossover != None:
            jmetal_algorithm_options['crossover']=crossover
        if selection != None:
            jmetal_algorithm_options['selection']=selection
        if termination_criterion != None:
            jmetal_algorithm_options['termination_criterion']=termination_criterion

        #self._create_termination_criterion()

        jmetal_algorithm_options.update(self._alg_options)
            
        self._algorithm = self._instantiate_jmetal_algorithm_object(jmetal_algorithm_options) #sto ako nema tog parametra mozda try catch, pa putem prepoznavanja koji argument ne postoji, ponovno pozvat kreiranje algoritma

        self._algorithm.run()

        sol = self._algorithm.get_result()
        
        self._sol = sol

        problem.evaluate(sol)

        opt_sol:OptimizationProblemSolution = callback_get_current_solution() #vraca objekt OptimizationProblemSolution koji je optbase objekt za cuvanje rjesenja u cistom numerickom obliku.

        solutions:List[OptimizationProblemSolution] = []

        solutions.append(opt_sol)

        return solutions
