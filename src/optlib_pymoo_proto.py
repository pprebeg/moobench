import numpy as np
from pymoo.core.problem import ElementwiseProblem
from optbase import OptimizationProblem,OptimizationAlgorithm,DesignVariable,DesignCriteria,DesignConstraint,OptimizationProblemSolution,DesignObjective
from typing import List, Dict
from pymoo.factory import get_algorithm, get_termination
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination, MultiObjectiveDefaultTermination



class WrappedPymooProblem(ElementwiseProblem):

    '''Class that inherits from ElementwiseProblem in order to instantiate an object that will define necessary methods for pymoo algorithm calculations. In the constructor - using super() function, 5 obligatory
    arguments are passed. Those are: how many design variables are there, how many objectives and constraints, as well as arrays of lower and upper boundaries. On top of that, two callback functions are passed. Those are
    called in _evaluate method. Crucial method that starts the calculation of problem.'''

    def __init__(self, desvars:List[DesignVariable], objs:List[DesignObjective], cons:List[DesignConstraint], xl:np.ndarray, xu:np.ndarray, callback_evaluate, callback_get_current_solution): #sve DesignVariable, DesignConstraint i ostalo pohranjeno je u OptimizationProblem klasi. Ako ce trebati bas Pymoo wrapperi tih klasa, onda se ovdje budu dodijelile.

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


##        #IZMJENA DIZAJNERSKIH PARAMETARA NA TEMELJU ONOG ŠTO JE DOBIVENO IZ PYMOO-A..
##        #x je ulazna vrijednost - algoritam daje x kao novi ulaz. Treba promijeniti vrijednosti x varijabli. Tako ocito radi algoritam. i treba se promijeniti u validan izlaz pozivom metode Analize_okvira
##        ix=0
##        for desvar in self._desvars:
##          desvar._dv.value=x[ix]
##          ix+=1
##
##        #POZIV proracuna kriterijskih funkcija - one ce pozvati proracun analyize metode AnalysisExecutora da bi se utvrdile nove vrijednosti modela i funkcija cilja i ogranicenja
##        #Dohvacanje tih vrijednosti podrazumijeva proracun koji se od tamo zove. On ce naravno promijeniti vrijednosti - parametara - pa posljedicno naprezanja i masa..
##        #ogranicenja i funkcije cilja naravno to dohvacaju putem callback konektora.
##        for obj in self._objs:
##            obj.criteria_fun_pymoo(x)
##        for con in self._cons:
##            con.criteria_fun_pymoo(x)

        
        self._callback_evaluate(x)
        
        solution=self._callback_get_current_solution() # u ovaj solution se naime spremaju u

        flist=[]
        glist=[]

        for i in range(self.n_obj):
            flist.append(solution.get_obj_value(i))               #numpy lista rjesenja
        print(flist)
        for i in range(self.n_constr):
            glist.append(solution.get_con_value(i))

        #out["F"] = np.column_stack(flist) #jednostavno - sprema numericke vrijednosti - svaki redak predstavlja jednorješenje, svaki stupac varijante iste varijable koje se istovremeno izracunavaju - toga nema kod ElementwiseProblem-a.
        out['F']=np.array(flist)

        #out["G"] = np.column_stack(glist)
        out['G']=np.array(glist)


class PymooOptimizationAlgorithm(OptimizationAlgorithm):

    '''Most important class of this interface. It takes settings as input, and creates a pymoo problem that is suitable for pymoo interface - function minimize. '''

    def __init__(self,method:str,alg_ctrl:Dict=None,term_ctrl:Dict=None):

        self._method=method
        self._alg_options=alg_ctrl
        self._term_ctrl=term_ctrl
        self._sol = None
        self.termination = None

        self._algorithm=get_algorithm(self._method,**self._alg_options) #dictionary self._options se raspakirava u keyword arguments. Nema greske ako je prazan dictionary..
                                                                            #self._opt_ctrl_iterables - ovo mozda nije prikladno! Tj. treba ipak prilagodba za algoritme pojedine.
                                                                            #Npr. RNSGA2 algoritam treba ref_points - tj. treba np.ndarray na drugom mjestu. al to samo taj algoritam, pa eto tome treba prilagoditi
        #Dodjela atributa za termination criterion! - ako moguce vise ih postaviti! onda elif zamijeniti sa if!
        if self._term_ctrl!=None:
            if self._term_ctrl.get('n_eval')!=None:
                value=self._term_ctrl.get('n_eval')
                self._termination=('n_eval',value)

            elif self._term_ctrl.get('n_gen')!=None:
                value=self._term_ctrl.get('n_gen')
                self._termination=('n_gen',value)

            elif self._term_ctrl.get('time')!=None:
                value=self._term_ctrl.get('time')
                self._termination=('time',value)


    @property
    def sol(self):

        return self._sol        # ovo ce biti ono sto vraca pymoo funkcija minimize.. ovdje - pymoo.Result

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

        #Algorithm
        alg = self._algorithm #ovaj bi objekt algorithm trebao instancirati korisnik u korisnickom programu kada definira algoritam optimizacije

        problem=WrappedPymooProblem(desvars, objectives, constraints, xl, xu, callback_evaluate, callback_get_current_solution) #ovaj callback_evaluate

        return problem

        #funkcija callback_evaluate

    def _generate_termination_criterion(self,condition):

        criterion=condition[0]
        value=condition[1]

        termination=get_termination(criterion,value)
        print(termination)

        return termination

    def _default_termination_criterion(self):

        if self._method=='nsga2':
            termination = MultiObjectiveDefaultTermination(
            x_tol=1e-8,
            cv_tol=1e-6,
            f_tol=0.0025,
            nth_gen=5,
            n_last=30,
            n_max_gen=1000,
            n_max_evals=100000)

##    elif self._method==
##    termination = SingleObjectiveDefaultTermination(
##    x_tol=1e-8,
##    cv_tol=1e-6,
##    f_tol=1e-6,
##    nth_gen=5,
##    n_last=20,
##    n_max_gen=1000,
##    n_max_evals=100000)

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
        for pymoo_dv in pymoo_dvs:
            x0.append(pymoo_dv.value)
        return x0

    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 callback_evaluate,callback_get_current_solution) -> List[OptimizationProblemSolution]:

        problem = self._generate_pymoo_problem(desvars, constraints, objectives, callback_evaluate, callback_get_current_solution)  #Definiranje pymoo problem

        if self.termination!=None:
            termination = self._generate_termination_criterion(self.termination)
        else:
            termination = self._default_termination_criterion()

        sol = minimize(problem, self._algorithm, termination, **self._alg_options) #ovo bi trebalo raditi baš kako treba! - termination=None, seed=None, verbose=False, display=None, callback=None, return_least_infeasible=False, save_history=False

        self._sol=sol #Sprema se pymoo.Result - klasa pymoo-a koja cuva rjesenje!

        opt_sol:OptimizationProblemSolution = callback_get_current_solution() #vraca objekt OptimizationProblemSolution koji je optbase objekt za cuvanje rjesenja u cistom numerickom obliku.

        solutions:List[OptimizationProblemSolution] = []

        solutions.append(opt_sol)

        return solutions
