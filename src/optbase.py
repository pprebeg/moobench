from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
from typing import List, Dict,TypeVar
from copy import copy, deepcopy
from utils import writecsv_listofstrings,readcsv_listofstrings,writecsv_dictionary,save_pareto_plot
import time
from datetime import datetime

class ConstrType(Enum): #Enumaratori za kontrolu tijeka programa
    GT = 1
    LT = 2
    EQ = 3

class AnalysisResultType(Enum): #isto za kontrolu tijeka dijelova programa
    OK = 0
    Error = -1
    Terminate_followers = 1

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

class CallbackGetSetConnector(BasicGetSetConnector): #jednostavno ce dohvacati odredjene funkcije koje ce predstavljati get i set funkcije value property-ja

    def __init__(self,cb_get,cb_set):
        self._cb_get = cb_get
        self._cb_set = cb_set

    @property
    def value(self):
        return self._cb_get()

    @value.setter
    def value(self,value):
        self._cb_set(value)

class CallbackGetConnector(BasicGetConnector): #Get connectori se koriste za vrijednosti kojih samo treba return vrijednosti - to su primjerice ogranicenja i funkcije cilja 

    def __init__(self,cb_get):
        self._cb_get = cb_get

    @property
    def value(self):
        return self._cb_get()

class NdArrayGetConnector(BasicGetConnector): #Get set konektori se koriste za vrijednosti koje treba i mijenjati - to su dizajnerske varijable. 
    def __init__(self, array: np.ndarray, ix: int):
        self._array = array
        self._ix = ix

    @property
    def value(self):
        return self._array[self._ix]

class NdArrayGetSetConnector(BasicGetSetConnector):         #putem ovih konektora - DesignVariable se mogu u potpunosti povezati sa array tipovima podataka. Intimno su povezani. Mijenjanjem varijable, mijenja se element array objekta
    def __init__(self, array: np.ndarray, ix: int):         # a ispisom varijable, ispisuje se element array-a. 
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
    def analyze(self)->AnalysisResultType:                     # u ovoj analyize funkciji mogu se definirati zavisnosti outarray-a o inarray-ju, važno primjetiti - nema argumenta inarray.. naime, on se mijenja tijekom optimizacije
        pass                                                    # u evaluate metodi OptimizationProblem klase..

class SimpleInputOutputArrayAnalysisExecutor(AnalysisExecutor): #jednostavni tip AnalysisExecutora - definiran je inarray kao lista svih ulaznih parametara (dizajnerski uglavnom) i izlaznih (vrijednosti funkcija ciljeva i ogranicenja)
    def __init__(self,num_inp,num_out):                         #te su bezlicne vrijednosti array objekta povezane sa DesignVariable, DesignConstraint i ostalim objektima putem 
        self.inarray = np.zeros(num_inp)                        #inicijalizira array-e
        self.outarray = np.zeros(num_out)


class DesignVariable(ABC):
    def __init__(self,name,connector:BasicGetSetConnector,lb=None,ub=None):
        self._name = name
        self._origin:BasicGetSetConnector = connector
        self._lb = lb
        self._ub = ub
        self._ndarray_connector: NdArrayGetConnector = None
        self._use_ndarray_connector: bool = False               #u optimize metodi OptimizationProblem, zove se vlastita metoda klase _init_opt_problem.. ona zove metodu - _set_ndarray_connector_to_components

    def set_ndarray_connector(self, ndarray_connector = None):  #poziva se u _set_ndarray_connector_to_components metodi - klasa OptimizationProblem... a to se poziva iz _init_opt_problem - iz optimize metode iste klase
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
    def value(self):                                    #vrijednost dizajnerske varijable - ako se koristi NdArrayConnector - vrijednost tog konektora. Ako ne, onda vrijednost BasicGetSetConnectora. 
        if self._use_ndarray_connector:
            return self._ndarray_connector.value
        else:
            return self._origin.value

    @property
    def origin_value(self) -> float:    #kao da zelis dobiti vrijednost konektora - DesignVariable.origin_value.value (to je vrijednost u konektoru)
        return self._origin.value

    @value.setter
    def value(self,value):
        self._origin.value = value

    def get_bounds(self):
        return self.lower_bound, self.upper_bound #ovo vraca tuple

    @property
    def status(self):
        curval = self.value #current value?
        if self.lower_bound is not None:
            if self.lower_bound > curval:
                return ' lb --'
            elif np.isclose(curval, self.lower_bound):
                return ' lb'
        elif self.upper_bound is not None:
            if self.upper_bound < curval:
                return ' ub --'
            elif np.isclose(curval,self.upper_bound):
                return ' ub'
        return  ''

    def get_info(self):
        return self.name+' = '+ str(self.origin_value)+self.status


class DesignCriteria(ABC):                                  #ima smisla naziv - kriteriji za dizajn su i ogranicenja i ciljevi! Tako je ovo apstraktna klasa. 
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
    def value_lt_0(self):                   #ovo je za PYMOO dodano.. 
        if self.con_type == ConstrType.GT:
            return -self.value + self.rhs   
        else:                               
            return self.value - self.rhs
        return
    
    @property
    def value_gt_0_normalized(self):                              #a/b > 4 ili a/b < 20
        if self.con_type == ConstrType.LT:
            return (self.rhs - self.value) / (self.value + self.rhs)    # (20 - a/b) / (20 + a/b)  primjer a/b = 10
        else: 
            return (self.value - self.rhs) / (self.value + self.rhs)    # (a/b - 4) / (4 + a/b) primjer a/b = 5

    @property
    def value_lt_0_normalized(self):
        if self.con_type == ConstrType.GT:
            return (self.value - self.rhs) / (self.value + self.rhs)
        else: 
            return (self.rhs - self.value) / (self.value + self.rhs)

    @property
    def status(self)->str:          #HOCE LI OVO RADITI ZA PYMOO KOJEM JE self.value_lt_0
        absrhs = np.abs(self._rhs)
        if absrhs > 0.0:
            if self.value_gt_0/absrhs < -0.001:
                return '--'
            elif self.value_gt_0/absrhs < 0.001:
                return '+-'
            else:
                return ''
        else:
            if self.value_gt_0 < -0.001:
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

class RatioGetCallbackConnector(BasicGetConnector):

    '''Class used to connect and normalize objective functions and such.'''

    def __init__(self, num_callback, denom_value:float):
        pass
        self._numerator_callback = num_callback
        self._denominator_value:float = denom_value

    @property
    def value(self):
        return self._numerator_callback()/self._denominator_value

class OptimizationProblemSolution():

    '''Class for holding optimization solutions - values of design variables, objective functions and constraints. Storing is envoked in Optimization Problem class. '''
    def __init__(self, num_var, num_obj,num_constr):
        self._dvs = np.zeros(num_var)
        self._objs = np.zeros(num_obj)
        self._cons = np.zeros(num_constr)
        self._num_var = num_var
        self._num_obj = num_obj
        self._num_con = num_constr
        self._values = np.zeros(self.num_opt_components)

    @property
    def num_var(self):
        return self._num_var
    @property
    def num_obj(self):
        return self._num_obj
    @property
    def num_con(self):
        return self._num_con
    @property
    def num_opt_components(self):
        return self._num_var + self._num_obj + self._num_con

    def get_var_ix(self, idv):
        return idv

    def get_obj_ix(self, iobj):
        return self._num_var+iobj

    def get_con_ix(self, icon):
        return self._num_var+self._num_obj+icon



    @property
    def objs(self):             #objective function solutions
        return self._values[self._num_var:self._num_var+self._num_obj]


    def get_obj_value(self,iobj:int):
        return self._values[self.get_obj_ix(iobj)]
    
    def set_variable_value(self, idv: int, value):
        self._values[self.get_var_ix(idv)] = value  # ovo bi smjela jedino metoda koja vrši proracun pozivati, kaj ne?

    def set_obj_value(self, iobj: int, value):
        self._values[self.get_obj_ix(iobj)] = value

    def set_con_value(self, icon: int, value):
        self._values[self.get_con_ix(icon)] = value

    def set_criteria_to_nan(self):
        self._values[self._num_var:self.num_opt_components] = np.nan

    def write_to_csvline(self):
        ar:List[str] = (np.char.mod('%f', self._values)).tolist()
        line = ",".join(ar)
        return line

    def read_from_csvline(self,line:str):
        self._values = np.fromstring(line, dtype=float, sep=',')


class OptimizationProblemMultipleSolutions(OptimizationProblemSolution):
    '''Class for holding optimization solutions - values of design variables, objective functions and constraints. Storing is envoked in Optimization Problem class. '''

    def __init__(self, num_var, num_obj, num_constr,num_solutions):
        super().__init__(num_var,num_obj,num_constr)
        self._num_sol = num_solutions
        self._values = np.zeros((self._num_sol,self.num_opt_components))

    @property
    def num_sol(self):
        return self._num_sol

    def get_ndarray_all_solutions_for_objective(self,iobj:int)->np.ndarray:
            return self._values[:,self.get_obj_ix(iobj)]

    def set_one_solution(self,isol:int,solution:OptimizationProblemSolution):
        self._values[isol,:] = solution._values[:]

    def set_variable_value(self, isol: int, idv: int, value):
        self._values[isol,self.get_var_ix(idv)] = value  # ovo bi smjela jedino metoda koja vrši proracun pozivati, kaj ne?

    def set_obj_value(self, isol: int, iobj: int, value):
        self._values[isol,self.get_obj_ix(iobj)] = value

    def set_con_value(self, isol: int, icon: int, value):
        self._values[isol,self.get_con_ix(icon)] = value

    def set_criteria_to_nan(self,isol:int):
        self._values[isol,self._num_var:self.num_opt_components] = np.nan

    def write_to_csvline(self)->List[str]:
        multiline = []
        for i in range(self._num_sol-1):
            ar: List[str] = (np.char.mod('%f', self._values[i,:])).tolist()
            multiline.append(",".join(ar)+'\n')
        ar: List[str] = (np.char.mod('%f', self._values[self._num_sol-1,:])).tolist()
        multiline.append(",".join(ar))
        return multiline

    def read_from_csvline(self, multiline: str):
        self._values = np.fromstring(multiline, dtype=float, sep=',')
        self._values.reshape(self.num_sol,self.num_opt_components)

class OptimizationAlgorithm(ABC): #doslovce klasa koja samo sluzi tome da bude nacrt tome kako ce ScipyOptimizationAlgorithm izgledati... 
    def __init__(self,name:str):
        self._name =name
        pass

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def sol(self):      #abstraktne metode moraju biti implementirane u klasama koje nasljedjuju od ove. 
        pass

    @abstractmethod
    def optimize(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,Callback_evaluate,Callback_get_curren_solution)->List[OptimizationProblemSolution]:
        pass

OO = TypeVar('OO', bound='OptimizationOutput')
class OptimizationOutput(ABC):
    def __init__(self,opt_alg_name:str,opt_problem_name:str,runtime:float,num_evaluations:int):
        self._creation_date_time:str = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        self._runtime:float = runtime
        self._opt_alg_name:str=opt_alg_name
        self._opt_problem_name: str = opt_problem_name
        self._num_evaluations = num_evaluations
        pass

    @property
    def num_var(self):
        return 0
    @property
    def num_obj(self):
        return 0
    @property
    def num_con(self):
        return 0

    @property
    def creation_date_time(self):
        return self._creation_date_time

    @property
    def runtime(self):
        return self._runtime
    @property
    def num_evaluations(self):
        return self._num_evaluations
    @property
    def opt_problem_name(self):
        return self._opt_problem_name

    @property
    def opt_alg_name(self):
        return self._opt_alg_name

    @property
    def full_name(self):
        return self.opt_problem_name+'_'+self.opt_alg_name

    @abstractmethod
    def get_solutions_string_list(self) -> List[str]:
        pass

    @abstractmethod
    def save_output(self,file_path,fieldnames):
        pass

    @classmethod
    def factory_from_out_file(out_file_path):
        pass

    def get_basic_data_dict(self)->Dict[str,str]:
        dd:Dict[str,str] = {}
        dd['opt_alg_name']=self.opt_alg_name
        dd['opt_problem_name'] = self.opt_problem_name
        dd['runtime'] = str(self.runtime)
        dd['num_evaluations'] = str(self.num_evaluations)
        dd['creation_date_time'] = self.creation_date_time
        return dd

    def get_info(self)->str:
        msg  =  'name: '+ self.full_name+'\n'
        msg +=  'date and time: '+ self.creation_date_time+'\n'
        msg +=  'runtime: '+ str(self.runtime)+'\n'
        msg +=  'num_evaluations: '+ str(self.num_evaluations)

        return msg

class SingleobjectiveOptimizationOutput(OptimizationOutput):
    def __init__(self,opt_alg_name:str,opt_problem_name:str,runtime:float,num_evaluations:int,solution:OptimizationProblemSolution):
        super().__init__(opt_alg_name,opt_problem_name,runtime,num_evaluations)
        self._solution: OptimizationProblemSolution = solution

    @classmethod
    def factory_from_out_file(cls, out_file_path):
        print('ERROR method not implemented!')

    def get_solutions_string_list(self) -> List[str]:
        return [self._solution.write_to_csvline() + '\n']

    def save_output(self, file_path,fieldnames):
        pass
    @property
    def num_var(self):
        return self._solution.num_var
    @property
    def num_obj(self):
        return self._solution.num_obj
    @property
    def num_con(self):
        return self._solution.num_con

class MultiobjectiveOptimizationOutput(OptimizationOutput):
    def __init__(self,opt_alg_name:str,opt_problem_name:str,runtime:float,num_evaluations:int,solutions:OptimizationProblemMultipleSolutions):
        super().__init__(opt_alg_name,opt_problem_name,runtime,num_evaluations)
        self._solutions: OptimizationProblemMultipleSolutions = solutions
        self._quality_measures:Dict[str,float] = {}

    def clear_quality_measures(self):
        self._quality_measures.clear()

    def add_quality_measure(self,key:str,value:float):
        self._quality_measures[key] =value

    @property
    def solutions(self):
        return self._solutions

    @property
    def num_sol(self):
        return self._solutions.num_sol
    @classmethod
    def factory_from_out_file(cls:type[OO], out_file_path):
        opt_alg_name:str= ''
        runtime:float = 0.0
        solutions: List[OptimizationProblemSolution] = []
        new_instance = cls(opt_alg_name,runtime,solutions)
        fieldnames_tocheck = ''
        return new_instance,fieldnames_tocheck

    def get_solutions_string_list(self) -> List[str]:
        return self._solutions.write_to_csvline()

    def save_pareto_plot(self, path, obj_1_name, obj_2_name):
        xdata = self.solutions.get_ndarray_all_solutions_for_objective(0)
        ydata = self.solutions.get_ndarray_all_solutions_for_objective(1)
        save_pareto_plot(path,self.full_name,xdata,ydata,obj_1_name,obj_2_name)

    def save_output(self, folder_path,fieldnames):
        file_path= folder_path+'\\'+self.full_name
        sols_file_path = file_path+'_s.csv'
        file_path+='.csv'
        sol_lines = self.get_solutions_string_list()
        writecsv_listofstrings(sols_file_path, fieldnames, sol_lines)
        dd = self.get_basic_data_dict()
        dd['num_sol'] = self.num_sol
        dd.update(self._quality_measures)
        dd['sols_file_path'] = sols_file_path
        writecsv_dictionary(file_path,dd)

    def append_output_data_dictionary(self,output_data:Dict[str,str]):
        pass
    def get_info(self)->str:
        msg  = super().get_info()
        msg  +=  '\nnum solutions: '+ str(self.num_sol)
        return msg
    @property
    def num_var(self):
        return self._solutions.num_var
    @property
    def num_obj(self):
        return self._solutions.num_obj
    @property
    def num_con(self):
        return self._solutions.num_con

class OptimizationProblem(ABC):                 #ovo je vrlo vazna klasa gdje je vecina funkcionalnosti implementirana
    def __init__(self,name='Optimization problem name'):
        self._name = name
        self._description = ''
        self._desvars:List[DesignVariable] = []
        self._constraints: List[DesignConstraint] = []
        self._objectives: List[DesignObjective] = []
        self._cur_sol:OptimizationProblemSolution = None
        self._evaluated_solutions:List[OptimizationProblemSolution]= []
        self._save_all_evaluated_solutions = False # set to True when  doing optimization of computationaly expensive functions
        self._opt_output:OptimizationOutput = None
        self._analysis_executors:List[AnalysisExecutor]=[]  
        self._use_ndarray_connectors = False            #ovo se nigdje ne dodjeljuje na True u simpleopt testu ni ovdje, a opet dohvaćanje vrijednosti i dalje radi.. I, radi dobro. Potvrdjeno. 
        self._opt_algorithm:OptimizationAlgorithm = None #dodjeljuje se kroz korisnicki program op.opt_algorithm = ... u primjeru - ScipyOptimizationAlgorithm
        self._evalcount = 0
        pass
    @property
    def opt_output(self):
        return self._opt_output
    @property
    def is_multiobjective(self):
        if len(self._objectives) > 1:
            return True
        return False

    @property
    def full_name(self):
        if self._opt_algorithm is None:
            return self.name+ '_Unassigned_opt_algorithm'
        else:
            return self.name + '_'+self._opt_algorithm.name

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def name(self):
        return self._name
    #implemented methods
    def optimize(self,x0= None):
        if self.opt_algorithm is not None: # u scipy primjeru to je spremljen objekt tipa ScipyOptimizationAlgorithm 
            self._evalcount = 0
            self._init_opt_problem() #ako je defaultno da se ne koriste, onda se instancira objekt tipa OptimizationProblem prilikom kojeg se tri arraya inicijaliziraju s nulama, dizajnerske varijable, ciljevi i ogranicenja
            if x0 is None:           #defaultno je None.. ako se pozove op.optimize().. no obicno se prilikom poziva OptimizationAlgorithm.optimize() u zagradama definiraju pocetne vrijednosti dizajnerskih varijabli.. 
                x0 = self.get_initial_design() #vraca pocetne x-eve na jedan od dva nacina.. A to je automatsko random kreiranje
            tStart = time.perf_counter()  # početak mjerenja vremena
            sols = self.opt_algorithm.optimize(self._desvars, #tu se zapravo poziva u ovom pocetnom slucaju (koji sam dobio od profesora) ScipyOptimizaionAlgorithm..
                 self._constraints,self._objectives, x0,        #self._desvars, self._constraints itd. - dodijeljene su preko metoda, preko korisnickih programa.. s onim add_constraint, add_design_variable.. itd. 
                 self.evaluate,self.get_current_sol)            #to su vlastite funkcije ove klase!!!!
            tEnd = time.perf_counter()
            dt = tEnd - tStart

            #Provjera da optlib (i algoritam nije vratio None)
            if sols != None:
                ne= self._evalcount
                if self.is_multiobjective:
                    mosols = OptimizationProblemMultipleSolutions(sols[0].num_var,sols[0].num_obj,sols[0].num_con,len(sols))
                    for i in range(len(sols)):
                        mosols.set_one_solution(i,sols[i])
                    self._opt_output = MultiobjectiveOptimizationOutput(self.opt_algorithm.name,self.name,dt,ne,mosols)
                else:
                    self._opt_output = SingleobjectiveOptimizationOutput(self.opt_algorithm.name,self.name, dt,ne, sols[-1])
                self.calculate_quality_measures()
            else:
                print(f'Algorithm {self._opt_algorithm.name} on problem {self._name} did not converge!')                
            return self.opt_algorithm.sol #prilikom optimize-a vraca konacno rjesenje optimizacije! Zato sto se zapravo pozivom linije sols = self.opt_algorithm.optimize zapravo poziva minimize iz scipy.optimize-a...

    def calculate_quality_measures(self):
        oo = self._opt_output
        if isinstance(oo,MultiobjectiveOptimizationOutput):
            # add all quality measures here
            oo.add_quality_measure('test_quality',42.42)
            
    def optimize_and_write(self,folder_path:str,x0= None):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string+" " + self.full_name + " optimization started")
        self.optimize(x0)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + " " + self.full_name + " optimization ended")
        path = folder_path
        try:
            self.write_output(path)
        except AttributeError:
            print('Optimization did not converge!')
            return False
        else:
            self.save_pareto_plot(path)
            return True
    

    def save_pareto_plot(self,path):
        out = self._opt_output
        if self.num_obj == 2 and isinstance(out,MultiobjectiveOptimizationOutput):
            out.save_pareto_plot(path,self._objectives[0].name, self._objectives[1].name)

    def write_output(self, path):
        fieldnames = self.get_dvobjcon_names_line()
        self._opt_output.save_output(path,fieldnames)

    def read_output(self,file_path):
        if self.is_multiobjective:
            opt_output, read_fieldnames = MultiobjectiveOptimizationOutput.factory_from_out_file(file_path)
        else:
            opt_output, read_fieldnames = SingleobjectiveOptimizationOutput.factory_from_out_file(file_path)
        fieldnames = self.get_dvobjcon_names_line()
        if read_fieldnames == fieldnames:
            self._opt_output = opt_output
            print('Compatible optimization output found and loaded for {}!'.format(self.full_name))
            return
        print('Warning: Compatible optimization output not found for {}!'.format(self.full_name))


    def get_current_sol(self):
        return self._cur_sol
    
    def evaluate(self,x:np.ndarray):
        for i in range (self.num_var):
            self._desvars[i].value = x[i]   #x-evi prosljedjeni u _evaluate funkciju (pymoo) update-aju DesignVariable 

        do_save = True
        for execu in self._analysis_executors:
            res = execu.analyze()
            self._evalcount+=1
            if res != (AnalysisResultType.OK):
                do_save = False
                break
        self.save_current_solution_variables()
        if do_save:
            self.save_current_solution_criteria()
        else:
            self._cur_sol.set_criteria_to_nan()
        # Intended use is for optimization of computationaly expensive functions
        if self._save_all_evaluated_solutions:
            self._evaluated_solutions.append(deepcopy(self._cur_sol))

    @property
    def num_var(self):
        return len(self._desvars)

    @property
    def num_con(self):
        return len(self._constraints)

    @property
    def num_obj(self):
        return len(self._objectives)

    def get_desvar(self,ix):
        return self._desvars[ix]

    @property
    def opt_algorithm(self):
        return self._opt_algorithm
    @property
    def solutions(self):
        return self._evaluated_solutions

    @opt_algorithm.setter
    def opt_algorithm(self, value):
        self._opt_algorithm = value

    def add_design_variable(self,dv):
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
        self._evaluated_solutions.clear()
        self._analysis_executors.clear()

    def _init_opt_problem(self):    #pozivanjem optimize metode objekta OptimizationAlgorithm klase, poziva se ova metoda inicijalizacije problema 
        self._cur_sol = OptimizationProblemSolution(self.num_var, self.num_obj, self.num_con) #Prilikom ovog poziva konstruktora OptimizationProblemSolution, stvaraju se tri prazna polja: za dizajnerske varijable, vrijednosti ciljeva i ogranicenja
        if self._use_ndarray_connectors:                #defaultno je na false! Vec se ovo definiralo u run_simpleopttestu.py, kako na true postavit?
            self._set_ndarray_connector_to_components() #Iskorištenje ndarray connectora koji su služili za uvijek jednako definiranje dizajnerskih varijabli, ciljeva i ogranicenja 

    def get_initial_design(self,is_random:bool=False):
        x0 = []                     #inicijalizira se x0 
        for dv in self._desvars:
            if is_random:                   #stvara random izmeðu granica x-eve.. ali dakle moraju imati dizajnerske varijable granice - ako nemaju greska... treba try-catch mozda.. 
                bnd = dv.get_bounds()
                x = np.random.rand() * (bnd[1] - bnd[0]) + bnd[0]
                x0.append(x)
            else:
                x0.append(dv.value) #moze biti da su definirane neke prve vrijednosti dizajnerskim varijablama... 
        return x0                   #funkcija vraca pocetni vektor dizajnerskih varijabli - x0

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

    def _set_ndarray_connector_to_components(self): #vrlo vazna metoda, u tri propertyja - num_var, num_obj i num_con spremljeni su broj potrebnih kreiranja poveznica putem konektora
        for i in range(self.num_var):
            cnct = NdArrayGetSetConnector(self._cur_sol._values,self._cur_sol.get_var_ix(i))  #kreira objekt, šalje mu iz sadašnjeg rješenja pojedinu dizajn varijablu i indeks.. indeksom redom prolazimo kroz _desvars listu varijabli..
            self._desvars[i].set_ndarray_connector(cnct) #za pojedinu dizajnersku varijablu u listi, postavlja konektor cnct. 
        for i in range(self.num_obj):
            cnct = NdArrayGetSetConnector(self._cur_sol._values, self._cur_sol.get_obj_ix(i))
            self._objectives[i].set_ndarray_connector(cnct)
        for i in range(self.num_con):
            cnct = NdArrayGetSetConnector(self._cur_sol._values, self._cur_sol.get_con_ix(i))
            self._constraints[i].set_ndarray_connector(cnct)

    def add_to_solutions(self,sol):
        self._evaluated_solutions.append(sol)

    def add_current_to_solutions(self):
        self.add_to_solutions(self._cur_sol)    #ovo s append nece raditi - jer append samo sprema pokazivac na tu vrijednost! Dodati barem deepcopy!

    def get_info(self):
        msg='------------ Optimization problem info -------------------\n'
        msg += 'Name: {0}\n'.format(self.name)
        if self.description != '':
            msg += '{0}\n'.format(self.description)
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

    def get_dvobjcon_names_line(self) -> List[str]:
        names = []
        for item in self._desvars:
            names.append(item.name)
        for item in self._objectives:
            names.append(item.name)
        for item in self._constraints:
            names.append(item.name)
        return  ",".join(names)+'\n'



