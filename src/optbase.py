from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
from typing import List, Dict

class ConstrType(Enum):
    GT = 1
    LT = 2
    EQ = 3

class AnalysisResultType(Enum):
    OK = 0
    Error = -1
    Terminate_folowers = 1

class BasicGetConnector():

    def __init__(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

class BasicGetSetConnector():

    def __init__(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self,value):
        pass

class CalbackGetSetConnector(BasicGetSetConnector):

    def __init__(self,cb_get,cb_set):
        self._cb_get = cb_get
        self._cb_set = cb_set

    @property
    def value(self):
        return self._cb_get()

    @value.setter
    def value(self,value):
        self._cb_set(value)

class CalbackGetConnector(BasicGetConnector):

    def __init__(self,cb_get):
        self._cb_get = cb_get

    @property
    def value(self):
        return self._cb_get()

class NdArrayGetConnector(BasicGetConnector):
    def __init__(self, array: np.ndarray, ix: int):
        self._array = array
        self._ix = ix

    @property
    def value(self):
        return self._array[self._ix]

class NdArrayGetSetConnector(BasicGetSetConnector):
    def __init__(self, array: np.ndarray, ix: int):
        self._array = array
        self._ix = ix

    @property
    def value(self):
        return self._array[self._ix]

    @value.setter
    def value(self, value):
        self._array[self._ix] = value

class AnalysisExecutor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def analyize(self)->AnalysisResultType:
        pass

class SimpleInputOutputArrayAnalysisExecutor(AnalysisExecutor):
    def __init__(self,num_inp,num_out):
        self.inarray = np.zeros(num_inp)
        self.outarray = np.zeros(num_out)


class DesignVariable(ABC):
    def __init__(self,name,connector:BasicGetSetConnector,lb=None,ub=None):
        self._name = name
        self._origin:BasicGetSetConnector = connector
        self._lb = lb
        self._ub = ub
        self._ndarray_connector: NdArrayGetConnector = None
        self._use_ndarray_connector: bool = False

    def set_ndarray_connector(self, ndarray_connector = None):
        self._ndarray_connector = ndarray_connector
        if ndarray_connector is not None:
            self._use_ndarray_connector = True
        else:
            self._use_ndarray_connector = False

    #abstract properties and methods


    #implemented properties and methods
    @property
    def name(self):
        return self._name
    @property
    def lower_bound(self)->float:
        return self._lb

    @lower_bound.setter
    def lower_bound(self, value):
        self._lb = value

    @property
    def upper_bound(self)->float:
        return self._ub

    @upper_bound.setter
    def upper_bound(self,value):
        self._ub = value

    @property
    def value(self):
        if self._use_ndarray_connector:
            return self._ndarray_connector.value
        else:
            return self._origin.value

    @property
    def origin_value(self) -> float:
        return self._origin.value

    @value.setter
    def value(self,value):
        self._origin.value = value

    def get_bounds(self):
        return self.lower_bound, self.upper_bound

    @property
    def status(self):
        curval = self.value #current value?
        if self.lower_bound > curval:
            return ' lb --'                     #izlazi su doslovce ovi stringovi - korisnik mora znati kaj znace
        elif self.upper_bound < curval:
            return ' ub --'
        elif np.isclose(curval,self.lower_bound):   #numpy funkcija ako su dva broja unutar tolerancije slicna - ili nizove more usporedjivati, element po element
            return ' lb'
        elif np.isclose(curval,self.upper_bound):
            return ' ub'
        return  ''

    def get_info(self):
        return self.name+' = '+ str(self.origin_value)+self.status


class DesignCriteria(ABC):
    def __init__(self,name,connector:BasicGetConnector):
        self._name = name
        self._origin:BasicGetConnector = connector
        self._ndarray_connector:NdArrayGetConnector = None
        self._use_ndarray_connector:bool = False

    def set_ndarray_connector(self, ndarray_connector=None):
        self._ndarray_connector = ndarray_connector
        if ndarray_connector is not None:
            self._use_ndarray_connector = True
        else:
            self._use_ndarray_connector = False

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        if self._use_ndarray_connector:
            return self._ndarray_connector.value
        else:
            return self._origin.value

    @property
    def origin_value(self)->float:
        return self._origin.value

    def get_info(self):
        return self.name+' = '+ str(self.origin_value)

class DesignObjective(DesignCriteria):                      #vjerojatno funkcija cilja, nasljedjuje kriterij? 
    def __init__(self,name,connector:BasicGetConnector):
        super(DesignObjective, self).__init__(name,connector)   
        pass

class DesignConstraint(DesignCriteria):
    def __init__(self,name,connector:BasicGetConnector,rhs:float=0.0,con_type:ConstrType = ConstrType.GT):
        super(DesignConstraint, self).__init__(name,connector)
        self._con_type = con_type #kakav je constraint, jednakost ili nejednakost, defaultni je greater then ocito
        self._rhs = rhs           #rhs je right hand side - ona vrijednost s desne strane jednakosti ili nejednakosti.
        pass

    @property
    def rhs(self):
        return self._rhs
    @rhs.setter
    def rhs(self, value):
        self._rhs = value

    @property
    def con_type(self):
        return self._con_type
    @con_type.setter
    def con_type(self, value):
        self._con_type = value

    @property
    def value_gt_0(self):
        if self.con_type == ConstrType.LT:
            return -self.value + self.rhs   # u ovoj se formi moraju napisati funkcije koje vracaju vrijednosti za ogranicenja
        else:                               #scipy samo na jedan nacin to moze primati, samo kao greater than vjerojatno, a ovo omogucuje i less than definiciju
            return self.value - self.rhs
        return

    @property
    def status(self)->str:
        absrhs = np.abs(self._rhs)
        if absrhs > 0.0:
            if self.value_gt_0/absrhs < -0.001:
                return '--'
            elif self.value_gt_0/absrhs < 0.001:
                return '+-'
            else:
                return ''
        else:
            if self.value_gt_0 < -0.01:
                return '--'
            elif self.value_gt_0 < 0.001:
                return '+-'
            else:
                return ''

    def get_info(self):
        if self._con_type == ConstrType.GT:
            tp = '>'
        elif self._con_type == ConstrType.LT:
            tp = '<'
        else:
            tp = '='
        return self.name+' = '+ str(self.origin_value) +' ' +tp+' '+ str(self._rhs)+'  '+self.status

class RatioDesignConnector(BasicGetConnector):

    def __init__(self, num:DesignVariable, denom:DesignVariable):
        pass
        self._numerator:DesignVariable = num
        self._denominator:DesignVariable = denom

    @property
    def value(self):
        return self._numerator.value/self._denominator.value

class OptimizationProblemSolution():
    def __init__(self, num_var, num_obj,num_constr):
        self._dvs = np.zeros(num_var)
        self._objs = np.zeros(num_obj)
        self._cons = np.zeros(num_constr)

    @property
    def dvs(self):              #design variable solutions
        return self._dvs

    @property
    def objs(self):             #objective function solutions
        return self._objs

    @property
    def cons(self):             #constraints
        return self._cons

    def get_variable_value(self,ix:int):
        return self._dvs[ix]    #vracanje vrijednosti odredjene design variable
    def set_variable_value(self,ix:int,value):
        self._dvs[ix] = value

    def get_obj_value(self,ix:int):
        return self._objs[ix]
    def set_obj_value(self,ix:int,value):
        self._objs[ix] = value

    def get_con_value(self,ix:int):
        return self._cons[ix]
    def set_con_value(self,ix:int,value):
        self._cons[ix] = value

    def set_criteria_to_nan(self):          #IEEE 754 floating point reprezentacija matematickog pojma NAN - neodredjeno 0/0 npr.
        self._objs[:] = np.nan
        self._cons[:] = np.nan

class OptimizationAlgorithm(ABC):
    def __init__(self,opt_ctrl):
        pass

    @property
    @abstractmethod
    def sol(self):      #abstraktne metode moraju biti implementirane u klasama koje nasljedjuju od ove. 
        pass

    @abstractmethod
    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,calback_evaluate,calback_get_curren_solution)->List[OptimizationProblemSolution]:
        pass

class OptimizationProblem(ABC):
    def __init__(self):
        self._desvars:List[DesignVariable] = []
        self._constraints: List[DesignConstraint] = []
        self._objectives: List[DesignObjective] = []
        self._cur_sol:OptimizationProblemSolution = None
        self._solutions:List[OptimizationProblemSolution]= []
        self._analysis_executors:List[AnalysisExecutor]=[]
        self._use_ndarray_connectors = False
        self._opt_algorithm:OptimizationAlgorithm = None
        pass


    #implemented methods
    def optimize(self,x0= None):
        if self.opt_algorithm is not None:
            self._init_opt_problem()
            if x0 is None:
                x0 = self.get_initial_design()
            sols = self.opt_algorithm.optimize(self._desvars,
                 self._constraints,self._objectives, x0,
                 self.evaluate,self.get_current_sol)
            self._solutions.extend(sols)
            return self.opt_algorithm.sol

    def get_current_sol(self):
        return self._cur_sol
    def evaluate(self,x:np.ndarray):
        for i in range (self.num_var):
            self._desvars[i].value = x[i]

        do_save = True
        for exec in self._analysis_executors:
            res = exec.analyize()
            if res != (AnalysisResultType.OK):
                do_save = False
                break
        self.save_current_solution_variables()
        if do_save:
            self.save_current_solution_criteria()
        else:
            self._cur_sol.set_criteria_to_nan()

    @property
    def num_var(self):
        return len(self._desvars)

    @property
    def num_con(self):
        return len(self._constraints)

    @property
    def num_obj(self):
        return len(self._objectives)

    @property
    def opt_algorithm(self):
        return self._opt_algorithm
    @property
    def solutions(self):
        return self._solutions

    @opt_algorithm.setter
    def opt_algorithm(self, value):
        self._opt_algorithm = value

    def add_design_varible(self,dv):
        self._desvars.append(dv)

    def add_constraint(self,item):
        self._constraints.append(item)

    def add_objective(self,item):
        self._objectives.append(item)

    def add_analysis_executor(self,item):
        self._analysis_executors.append(item)

    def clear_problem_data(self):
        self._desvars.clear()
        self._constraints.clear()
        self._objectives.clear()
        self._solutions.clear()
        self._analysis_executors.clear()

    def _init_opt_problem(self):
        self._cur_sol = OptimizationProblemSolution(self.num_var, self.num_obj, self.num_con)
        if self._use_ndarray_connectors:
            self._set_ndarray_connector_to_components()

    def get_initial_design(self,is_random:bool=False):
        x0 = []
        for dv in self._desvars:
            if is_random:
                bnd = dv.get_bounds()
                x = np.random.rand() * (bnd[1] - bnd[0]) + bnd[0]
                x0.append(x)
            else:
                x0.append(dv.value)
        return x0

    def get_ub_design(self,is_random:bool=False):
        x0 = []
        for dv in self._desvars:
            bnd = dv.get_bounds()
            x0.append(bnd[1])
        return x0

    def save_current_solution_criteria(self):
        for i in range(self.num_obj):
            self._cur_sol.set_obj_value(i, self._objectives[i].origin_value)
        for i in range(self.num_con):
            self._cur_sol.set_con_value(i, self._constraints[i].origin_value)

    def save_current_solution_variables(self):
        for i in range(self.num_var):
            self._cur_sol.set_variable_value(i, self._desvars[i].origin_value)


    def _set_ndarray_connector_to_components(self):
        for i in range(self.num_var):
            cnct = NdArrayGetSetConnector(self._cur_sol.dvs,i)
            self._desvars[i].set_ndarray_connector(cnct)
        for i in range(self.num_obj):
            cnct = NdArrayGetSetConnector(self._cur_sol.objs, i)
            self._objectives[i].set_ndarray_connector(cnct)
        for i in range(self.num_con):
            cnct = NdArrayGetSetConnector(self._cur_sol.cons, i)
            self._constraints[i].set_ndarray_connector(cnct)

    def add_to_solutions(self,sol):
        self._solutions.append(sol)

    def add_current_to_solutions(self):
        self.add_to_solutions(self._cur_sol)

    def get_info(self):
        msg='------------ Optimization problem info -------------------\n'
        msg+= 'Num variables: {0}\n'.format(self.num_var)
        for item in self._desvars:
            msg+=item.get_info()+'\n'
        msg += 'Num constraints: {0}\n'.format(self.num_con)
        for item in self._constraints:
            msg += item.get_info() + '\n'
        msg += 'Num objectives: {0}\n'.format(self.num_obj)
        for item in self._objectives:
            msg += item.get_info() + '\n'
        msg += '-------------------------------------------------------'
        return msg

