from abc import ABC,abstractmethod
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from optbase import OptimizationProblem,OptimizationAlgorithm,DesignVariable,DesignCriteria,DesignConstraint,OptimizationProblemSolution,DesignObjective
from typing import List, Dict
from pymoo.core.result import Result
from pymoo.factory import get_algorithm, get_sampling, get_selection, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination, MultiObjectiveDefaultTermination
from copy import copy, deepcopy



class PymooConstraint():
    def __init__(self, con:DesignConstraint):    #tu bi se trebalo kod constrainta koristiti ono kaj je moja ideja bila na seminarskom.. con mora biti funkcija  
        self._criteria:DesignConstraint=con

    @property
    def con(self)->DesignConstraint:
        return self._criteria

    def get_con_value(self) ->float: 
        return self.con.value_lt_0 

class WrappedPymooProblem(ElementwiseProblem):

    '''Class that inherits from ElementwiseProblem in order to instantiate an object that will define necessary methods for pymoo algorithm calculations. In the constructor - using super() function, 5 obligatory
    arguments are passed. Those are: how many design variables are there, how many objectives and constraints, as well as arrays of lower and upper boundaries. On top of that, two callback functions are passed. Those are
    called in _evaluate method. Crucial method that starts the calculation of problem.'''

    def __init__(self, desvars:List[DesignVariable], objs:List[DesignObjective], cons:List[PymooConstraint], xl:np.ndarray, xu:np.ndarray, callback_evaluate, callback_get_current_solution): #sve DesignVariable, DesignConstraint i ostalo pohranjeno je u OptimizationProblem klasi. Ako ce trebati bas Pymoo wrapperi tih klasa, onda se ovdje budu dodijelile.

        self._callback_evaluate=callback_evaluate
        self._callback_get_current_solution=callback_get_current_solution

        self.n_var = len(desvars)
        self.n_obj = len(objs)
        self.n_constr = len(cons)
        self._desvars=desvars
        self._cons=cons
        self._objs=objs

        super().__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=xl, xu=xu) #inicijalizacija WrappedPymooProblem2 kao child klase ElementWiseProblem-a


    def _evaluate(self, x:np.ndarray, out, *args, **kwargs):
        
        self._callback_evaluate(x)
        
        solution:OptimizationProblemSolution=self._callback_get_current_solution() # OptimizationProblemSolution u ovaj solution se naime spremaju u

        flist=[]
        glist=[]

        for i in range(self.n_obj):
            flist.append(solution.get_obj_value(i))               #numpy lista rjesenja

        for i in range(self.n_constr):
            glist.append(self._cons[i].get_con_value())

        out['F']=np.array(flist)
        
        out['G']=np.array(glist)


class PymooOptimizationAlgorithm(OptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a pymoo problem that is suitable for pymoo interface - function minimize. '''

    def __init__(self,method:str,alg_ctrl:Dict=None, minimize_ctrl:Dict = None):    #ovo neka ostane za pymoo specificno! da se jos moze poslati jedan dictionary koji dodatno kontrolira minimize funkciju! 

        self._method=method
        self.termination_generated = False

        mutation = None
        crossover = None
        sampling = None
        selection = None
        termination = None

        self._alg_options = {}
        self._minimize_ctrl = {}

        self._sol:Result = None
        self._optimal_solutions: List[OptimizationProblemSolution] = []
        
        mutation = alg_ctrl.get('mutation')
        if mutation != None:
            alg_ctrl.pop('mutation')
            mutation_obj = self._generate_mutation(mutation)
            self._alg_options['mutation'] = mutation_obj
            
        crossover = alg_ctrl.get('crossover')
        if crossover != None:
            alg_ctrl.pop('crossover')
            print(crossover)
            crossover_obj = self._generate_crossover(crossover)
            self._alg_options['crossover'] = crossover_obj
            
        sampling = alg_ctrl.get('sampling')
        if sampling != None:
            alg_ctrl.pop('sampling')
            sampling_obj = self._generate_sampling(sampling)
            self._alg_options['sampling'] = sampling_obj
            
        selection = alg_ctrl.get('selection')
        if selection != None:
            alg_ctrl.pop('selection')
            selection_obj = self._generate_selection(selection)
            self._alg_options['selection'] = selection_obj
            
        termination = alg_ctrl.get('termination')
        if termination != None:
            alg_ctrl.pop('termination')
            self._termination = self._generate_termination(termination)
            self.termination_generated = True

        
        self._alg_options.update(alg_ctrl)
        
        self._algorithm=get_algorithm(self._method,**self._alg_options)     #dictionary self._options se raspakirava u keyword arguments. Nema greske ako je prazan dictionary..
                                                                            #Npr. RNSGA2 algoritam treba ref_points - tj. treba np.ndarray na drugom mjestu. al to samo taj algoritam, pa eto tome treba prilagoditi
        if minimize_ctrl != None:
            self._minimize_ctrl.update(minimize_ctrl)

        
    @property
    def sol(self) -> Result:

        return self._sol      # ovo ce biti ono sto vraca pymoo funkcija minimize.. ovdje - pymoo.Result

    def _generate_pymoo_problem(self,
                desvars: List[DesignVariable],
                constraints: List[DesignConstraint],
                objectives: List[DesignObjective],
                callback_evaluate,
                callback_get_current_solution)-> WrappedPymooProblem:

        '''This function is crucial to transform general problem, algorithm definitions, variables, etc. defined through optbase.py. into pymoo specific problem, algorithm and so on. This, and only this type of data
            can than be passed through to pymoo. '''


        # Design Variables and Bounds
        xl: List[float] = []
        xu: List[float] = []

        xl, xu=self._get_bounds(desvars)    #kreiranje niza za donje i gornje granice varijabli

        #Constraints
        pymoo_cons:List[PymooConstraint]=[]
        for con in constraints:
            pymoocon = PymooConstraint(con)
            pymoo_cons.append(pymoocon)

        #Algorithm
        alg = self._algorithm #Instancira se u __init__

        problem=WrappedPymooProblem(desvars, objectives, pymoo_cons, xl, xu, callback_evaluate, callback_get_current_solution) #ovaj callback_evaluate

        return problem
    
    def _generate_crossover(self, item):

        type_of_crossover:str = item.get('name')
        item.pop('name')

        crossover = get_crossover(type_of_crossover, **item)
        

        return crossover

    def _generate_mutation(self, item:Dict):

        type_of_mutation:str = item.get('name')
        item.pop('name')
        print(item)
        mutation = get_mutation(type_of_mutation, **item)

        return mutation

    def _generate_selection(self, item):

        type_of_selection:str = item.get('name')
        item.pop('name')

        selection = get_selection(type_of_selection, **item)

        return selection

    def _generate_termination(self, item):

        type_of_termination:str = item[0]
        value:int = item[1]

        termination = get_termination(type_of_termination, value)

        return termination

    def _get_bounds(self, dvs:List[DesignVariable]):

        '''Method that extracts bounds from a list of DesignVariable objects. It returns two numpy arrays in a tuple. First element is a list of lower bounds, while second is a list of upper bounds.'''

        lbs = []
        ubs = []
        for dv in dvs:
            lb,ub = (dv.get_bounds())
            lbs.append(lb)
            ubs.append(ub)

        return np.array(lbs), np.array(ubs)

    def _get_current_desvar_values(self, dvs:List[DesignVariable]):
        x0 = []
        for pymoo_dv in dvs:
            x0.append(pymoo_dv.value)
        return x0
    
    @abstractmethod
    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        pass

class PymooOptimizationAlgorithmMulti(PymooOptimizationAlgorithm):

    def __init__(self, method:str,alg_ctrl:Dict=None, minimize_ctrl:Dict = None):
        super().__init__(method,alg_ctrl, minimize_ctrl)

        if not self.termination_generated:
            self._termination = self._default_termination_criterion()
            
    def _default_termination_criterion(self):

        termination = MultiObjectiveDefaultTermination(
        x_tol=1e-8,
        cv_tol=1e-6,
        f_tol=0.0025,
        nth_gen=5,
        n_last=30,
        n_max_gen=1000,
        n_max_evals=100000)

        return termination

    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        #GENERATING PYMOO PROBLEM

        problem = self._generate_pymoo_problem(desvars, constraints, objectives, callback_evaluate, callback_get_current_solution)        

        #CONDUCTING PYMOO OPTIMIZATION
            
        sol = minimize(problem, self._algorithm, self._termination, **self._minimize_ctrl)

        self._sol = sol #Sprema se pymoo.Result - klasa pymoo-a koja cuva optimalno i izvedivo (ako je dobiveno) rjesenje!

        #FINAL EVALUATION OF OPTIMAL SOLUTION TO BE STORED AS OptimizationProblemSolution
        
        x = sol.X
            
        solutions:List[OptimizationProblemSolution] = []
        
        for index, x_individual in enumerate(x):
            out={}
            out['F'] = sol.F[index]
            out['G'] = sol.G[index]
            problem._evaluate(x_individual,out)     #ovo sprema OptimizationProblemSolution u OptimizationProblem
            opt_sol:OptimizationProblemSolution = callback_get_current_solution() #vraca OptimizationProblemSolution koji je optbase objekt za cuvanje rjesenja u numerickom obliku. ili izraditi copy objekta - funckionalnost na razini pymoo_proto.. ili u optbase prosiriti poziv ovog optimize-a sa jos jednim callbackom koji bi pozivao add_
            solutions.append(deepcopy(opt_sol))   #append adds a reference only! in solutions, there are just pointers to opt_sol! it should actually make copies that are independent one of another, so that it doesn't change when opt_sol change                
    
        return solutions  

class PymooOptimizationAlgorithmSingle(PymooOptimizationAlgorithm):

    def __init__(self, method:str,alg_ctrl:Dict=None, minimize_ctrl:Dict = None):
        super().__init__(method, alg_ctrl, minimize_ctrl)

        if not self.termination_generated:
            self._termination = self._default_termination_criterion()

    def _default_termination_criterion(self):

        termination = SingleObjectiveDefaultTermination(
        x_tol=1e-8,
        cv_tol=1e-6,
        f_tol=1e-6,
        nth_gen=5,
        n_last=20,
        n_max_gen=1000,
        n_max_evals=100000)

        return termination

    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        #GENERATING PYMOO PROBLEM

        problem = self._generate_pymoo_problem(desvars, constraints, objectives, callback_evaluate, callback_get_current_solution)

        #CONDUCTING PYMOO OPTIMIZATION
            
        sol = minimize(problem, self._algorithm, self._termination, **self._minimize_ctrl)

        self._sol = sol #Tipa pymoo.Result - klasa pymoo-a koja cuva optimalno i izvedivo (ako je dobiveno) rjesenje!

        #FINAL EVALUATION OF OPTIMAL SOLUTION TO BE STORED AS OptimizationProblemSolution
        #PRIVREMENO - RAZDVOJITI I DA NASLJEDJUJU
        #print(sol)
        x = sol.X

        if x.all == None:
            print('Algoritam nije uspio naci izvedivo rjesenje!')
            return
        print(x)

        out={}
        out['F'] = sol.F
        out['G'] = sol.G
        problem._evaluate(x,out)

        #STORING

        opt_sol:OptimizationProblemSolution = callback_get_current_solution()
        solutions:List[OptimizationProblemSolution] = []
        solutions.append(deepcopy(opt_sol))
        
        return solutions
  
