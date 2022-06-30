from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
from typing import List, Dict
from copy import copy, deepcopy

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
    def analyize(self)->AnalysisResultType:                     # u ovoj analyize funkciji mogu se definirati zavisnosti outarray-a o inarray-ju, važno primjetiti - nema argumenta inarray.. naime, on se mijenja tijekom optimizacije
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
        if self.lower_bound > curval:           #izasao ispod donje granice
            return ' lb --'                     #izlazi su doslovce ovi stringovi - korisnik mora znati kaj znace
        elif self.upper_bound < curval:         #iznad gornje granice
            return ' ub --'
        elif np.isclose(curval,self.lower_bound):   #na donjoj granici - numpy funkcija ako su dva broja unutar tolerancije slicna - ili nizove more usporedjivati, element po element
            return ' lb'
        elif np.isclose(curval,self.upper_bound):   #na gornjoj granici
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
        return self._dvs[ix]    #vracanje vrijednosti odredjene design variable - indeks iz konektora vjerojatno ekstrahiramo.. pa proslijedimo ovom array-u? Mogu li se pobrkati ako nisu redom indeksi? 
    
    def set_variable_value(self,ix:int,value):
        self._dvs[ix] = value                   #ovo bi smjela jedino metoda koja vrši proracun pozivati, kaj ne? 

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

class OptimizationAlgorithm(ABC): #doslovce klasa koja samo sluzi tome da bude nacrt tome kako ce ScipyOptimizationAlgorithm izgledati... 
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
                 x0:np.ndarray,Callback_evaluate,Callback_get_curren_solution)->List[OptimizationProblemSolution]:
        pass

class OptimizationProblem(ABC):                 #ovo je vrlo vazna klasa gdje je vecina funkcionalnosti implementirana
    def __init__(self):
        self._desvars:List[DesignVariable] = []
        self._constraints: List[DesignConstraint] = []
        self._objectives: List[DesignObjective] = []
        self._cur_sol:OptimizationProblemSolution = None
        self._solutions:List[OptimizationProblemSolution]= []
        self._analysis_executors:List[AnalysisExecutor]=[]  
        self._use_ndarray_connectors = False            #ovo se nigdje ne dodjeljuje na True u simpleopt testu ni ovdje, a opet dohvaćanje vrijednosti i dalje radi.. I, radi dobro. Potvrdjeno. 
        self._opt_algorithm:OptimizationAlgorithm = None #dodjeljuje se kroz korisnicki program op.opt_algorithm = ... u primjeru - ScipyOptimizationAlgorithm
        pass


    #implemented methods
    def optimize(self,x0= None):
        if self.opt_algorithm is not None: # u scipy primjeru to je spremljen objekt tipa ScipyOptimizationAlgorithm 
            self._init_opt_problem() #ako je defaultno da se ne koriste, onda se instancira objekt tipa OptimizationProblem prilikom kojeg se tri arraya inicijaliziraju s nulama, dizajnerske varijable, ciljevi i ogranicenja
            if x0 is None:           #defaultno je None.. ako se pozove op.optimize().. no obicno se prilikom poziva OptimizationAlgorithm.optimize() u zagradama definiraju pocetne vrijednosti dizajnerskih varijabli.. 
                x0 = self.get_initial_design() #vraca pocetne x-eve na jedan od dva nacina.. A to je automatsko random kreiranje
            sols = self.opt_algorithm.optimize(self._desvars, #tu se zapravo poziva u ovom pocetnom slucaju (koji sam dobio od profesora) ScipyOptimizaionAlgorithm.. 
                 self._constraints,self._objectives, x0,        #self._desvars, self._constraints itd. - dodijeljene su preko metoda, preko korisnickih programa.. s onim add_constraint, add_design_variable.. itd. 
                 self.evaluate,self.get_current_sol)            #to su vlastite funkcije ove klase!!!!
            self._solutions.extend(sols) #extend metoda samo jednostavno spaja dvije liste ili dva iterabla u jedan. To je lista OptimizationProblemSolution
            return self.opt_algorithm.sol #prilikom optimize-a vraca konacno rjesenje optimizacije! Zato sto se zapravo pozivom linije sols = self.opt_algorithm.optimize zapravo poziva minimize iz scipy.optimize-a... 

    def get_current_sol(self):
        return self._cur_sol
    
    def evaluate(self,x:np.ndarray):
        for i in range (self.num_var):
            self._desvars[i].value = x[i]   #x-evi prosljedjeni u _evaluate funkciju (pymoo) update-aju DesignVariable 

        do_save = True
        for execu in self._analysis_executors:
            res = execu.analyize()
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

    def get_desvar(self,ix):
        return self._desvars[ix]

    @property
    def opt_algorithm(self):
        return self._opt_algorithm
    @property
    def solutions(self):
        return self._solutions

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
        self._solutions.clear()
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
            cnct = NdArrayGetSetConnector(self._cur_sol.dvs,i)  #kreira objekt, šalje mu iz sadašnjeg rješenja pojedinu dizajn varijablu i indeks.. indeksom redom prolazimo kroz _desvars listu varijabli.. 
            self._desvars[i].set_ndarray_connector(cnct) #za pojedinu dizajnersku varijablu u listi, postavlja konektor cnct. 
        for i in range(self.num_obj):
            cnct = NdArrayGetSetConnector(self._cur_sol.objs, i)
            self._objectives[i].set_ndarray_connector(cnct)
        for i in range(self.num_con):
            cnct = NdArrayGetSetConnector(self._cur_sol.cons, i)
            self._constraints[i].set_ndarray_connector(cnct)

    def add_to_solutions(self,sol):
        self._solutions.append(sol)         

    def add_current_to_solutions(self):
        self.add_to_solutions(self._cur_sol)    #ovo s append nece raditi - jer append samo sprema pokazivac na tu vrijednost! Dodati barem deepcopy!

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
        return print(msg)

