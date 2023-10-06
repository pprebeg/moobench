import numpy as np                      #optlib_scipy je zapravo sucelje izmedju modela i scipy-ja.. optbase kojim se koristi ima neke osnovne klase, funkcije, ajmo rec postavljen princip, a onda ovaj modul to prilagodjava scipyu
from abc import ABC,abstractmethod
from typing import List, Dict
from enum import Enum

try:
    from scipy.optimize import minimize
except ImportError:
    print('Import error: Scipy not installed')
is_optbase_imported = False                     #varijabla is_optbase_imported, importiranje modula optbase
try:                                            #pojedinacno importanje je odlicno jer ces dobiti gresku ako koja vazna klasa nije uvezena!
    from optbase import ConstrType, AnalysisResultType, AnalysisExecutor
    from optbase import DesignVariable, DesignObjective, DesignConstraint, OptimizationAlgorithm
    from optbase import DesignCriteria,OptimizationAlgorithm, OptimizationProblemSolution
    is_optbase_imported = True
except ImportError:
    pass
if not is_optbase_imported: #not False je true - dakle ako je false, nije uvezeno, pa treba probati drugi folder
    try:
        from femdir.optbase import ConstrType,AnalysisResultType,AnalysisExecutor
        from femdir.optbase import DesignVariable,DesignObjective,DesignConstraint,OptimizationAlgorithm
        from femdir.optbase import DesignCriteria,OptimizationAlgorithm, OptimizationProblemSolution
        is_optbase_imported = True
    except ImportError:
        pass

class ScipyModelAnalysis():
    def __init__(self,calback_analyse):
        self._rtol = 1e-10                      #postavljanje tolerancija za granice izgleda, vidi klasu DesignConstraint, property status
        self._atol = self._rtol * 1e-3
        self._xlast:np.ndarray = None           #to je samo najavljeno numpy niz.. ali je None
        self._calback_analyse_method = calback_analyse #funkcija ili metoda koja se kao callback salje 

    def init_analysis(self,num_var):
        self._xlast: np.ndarray = np.ones(num_var)*np.random.random() #random odabir zadnjeg x-a.. no mislim da to nije pocetni x, vec neki x u odnosu na koji ce se prvi put ova metoda koristiti _is_new_x_vals

    def _is_new_x_vals(self,x:np.ndarray):
        if np.allclose(x,self._xlast,rtol=self._rtol, atol=self._atol): #zanimljiva funkcija koja provjerava jesu li se promijenile design varijable
            return False
        return True

    def analyse(self,x:np.ndarray):
        if self._is_new_x_vals(x):      #jako sazet kod - tak bi i ja trebal svoj stari kod promijenit, za sve postoji funkcija
            self._calback_analyse_method(x) # Callback metoda koja se poziva je zapravo OptimizationProblem.evaluate metoda! A, ona zapravo pokazuje na analyize metodu u kojoj smo postavili problem.. drugim rijecima outarray iz inarray-a
            self._xlast[:]=x[:]         #ovo bi vjerojatno znacilo - stari postaju novi
        pass


class ScipyVariable():
    def __init__(self, dv:DesignVariable):  #samo preslikava vrijednost na ScipyVariablu u init funkciji, ali prosiruje
        self._dv:DesignVariable = dv        #DesignVariablu s mnogim drugim funkcijama.. Design varijabla zapravo unutar sebe ma mnogeprivatne atribute, definirane kao propertye aribute
        pass

    def get_bounds(self):
        return self._dv.get_bounds()
    
    @property
    def value(self):
        return self._dv.origin_value

    # Scipy minimize related methods
    def fun_COBYLA_lower_bound_constraint(self,x: np.ndarray):
        return self._dv.value - self._dv.lower_bound


    def fun_COBYLA_upper_bound_constraint(self,x: np.ndarray):
        return -self._dv.value + self._dv.upper_bound

    def get_lb_COBYLA_constraint_data(self):
        if self._dv.lower_bound is not None:
            return {'type': 'ineq', 'fun': self.fun_COBYLA_lower_bound_constraint}
        return None

    def get_ub_COBYLA_constraint_data(self):
        if self._dv.upper_bound is not None:
            return {'type': 'ineq', 'fun': self.fun_COBYLA_upper_bound_constraint}
        return None

class ScipyCriteria():
    def __init__(self, spa:ScipyModelAnalysis,crit:DesignCriteria):
        self._spa:ScipyModelAnalysis = spa
        self._criteria:DesignCriteria = crit
        pass

    def get_value_scipy(self)->float:
        return self._criteria.value

    def criteria_fun_scipy(self, x:np.ndarray)->float:
        self._spa.analyse(x)                #salju se kao argumenti - design varijable vjerojatno, ovi x-evi.. kao brojevi.. spa je scipy model analysis...
        return self.get_value_scipy()

class ScipyConstraint(ScipyCriteria):
    def __init__(self, spa:ScipyModelAnalysis,con:DesignConstraint):    #tu bi se trebalo kod constrainta koristiti ono kaj je moja ideja bila na seminarskom.. con mora biti funkcija  
        super().__init__(spa,con)

    @property
    def con(self)->DesignConstraint:
        return self._criteria

    def get_constraint_data_scipy(self):
        if self.con.con_type == ConstrType.EQ:
            return {'type': 'eq', 'fun': self.criteria_fun_scipy} #ovo je pokazivac na funkciju - callback.. jer nema zagrada!
        else:                                                     #da su zagrade (x), onda bi se u tom trenu pozvala funkcija, ovo je samo pokazivac!
            return {'type': 'ineq', 'fun': self.criteria_fun_scipy}

    def criteria_fun_scipy(self, x:np.ndarray) ->float: 
        self._spa.analyse(x)            #VAŽNO: ovo je analyse metoda ScipyModelAnalysis klase, a ne analyize metoda AnalysisExecutor klase!
        return self.con.value_gt_0      #EKSTREMNO VAZNO! iako pokrece analyse gdje se izracunavaju sve funkcije cilja i sva ogranicenja (SAMO AKO JE NOVI X!), vraca vrijednost samo
                                        #od tog ogranicenja! I, na taj nacin je, iako objedinjeno u analyse metodi ScipyModelAnalysis klase, ovdje to razluceno na svaku zasebe funkcija ogranicenja! redak self.con.value_gt_0 vraca jedan broj, vrijednost jedne funkcije ogranicenja
class ScipyObjective(ScipyCriteria):    #objective nasljedjuje kriterij - funkcija super, salju se spa i obj. 
    def __init__(self, spa:ScipyModelAnalysis,obj:DesignObjective):
        super().__init__(spa,obj)



class ScipyOptimizationAlgorithm(OptimizationAlgorithm):
    def __init__(self,name:str,method:str, opt_ctrl:Dict ):
        super().__init__(name)
        self._method = method
        if self._method is None:
            self._method = 'SLSQP'  #default
        self._options = {}
        self._sol = None
        for key,value in opt_ctrl.items():
            self._options[key] = value

    @property
    def sol(self):
        return self._sol

    def _generate_scipy_problem(self,desvars: List[DesignVariable],
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 calback_evaluate):        #ovo konacno pokazuje na funkciju evaluate u OptimizationProblem.optimize

        sp_ma:ScipyModelAnalysis = ScipyModelAnalysis(calback_evaluate) #VAŽNA LINIJA - kreira se taj ScipyModelAnalysis objekt - i prosljedjuje se pokazivac na funkciju - vidi poziv _generate_scipy_problem iz optimize metode ove klase
        # Design Variables
        sp_desvars: List[ScipyVariable] = []
        for dv in desvars:              #desvars bi trebala biti lista elemenata tipa DesignVariable... 
            spdv = ScipyVariable(dv)    #uzima DesignVariable i pretvara u ScipyVariable
            sp_desvars.append(spdv)
        # Constraints
        sp_cons:List[ScipyConstraint]=[]
        for con in constraints:
            spcon = ScipyConstraint(sp_ma,con)
            sp_cons.append(spcon)
        # Objective
        sp_obj:ScipyObjective = ScipyObjective(sp_ma,objectives[0])    #s obzirom da je jednociljno, samo je objectives[0].. da je viseciljnoo - bilo bi isto ko i za ogranicenja odmah iznad.. bila bi for petlja, koja bi presla
                                                                       #cijelu listu ogranicenja kao iterable.. for obj in objs npr.. 
        consdata = self._get_constraint_data(sp_cons)
        if self._method == 'COBYLA':
            bnds=None                       #COBYLA ide sve preko constraints-a
            dvcdata = self._get_bounds_as_constraint_data(sp_desvars) #funkcija specijalno za prevodjenje boundsa u constraints... 
            consdata.extend(dvcdata)
        else:
            bnds = self._get_bounds(sp_desvars) #saljes ScipyVariable, ova funkcija vraca bounds tako da ih ekstrahira iz ScipyVarijabli, tj. iz niza redom.. 
        objfun = sp_obj.criteria_fun_scipy      #Kao sto je za ogranicenja trebao consdata=_get_constraint_data(sp_cons), ovdje je jedan cilj optimizacije, i zato se ovako eksplicitno dodjeljuje - da vrati objfun - dakle callable - metodu ScipyObjective objekta/klase...
        sp_ma.init_analysis(len(desvars))       #ova linija ne daje x0 vec inicijalizira neki zadnji x - kao da je to zadnje rjesenje, iako x0 salje kao pocetni. 
        return (objfun,consdata,bnds)

    def _get_bounds(self, sp_dvs:List[ScipyVariable]):
        bnds = []
        for spdv in sp_dvs:
            bnds.append(spdv.get_bounds())
        return tuple(bnds)

    def _get_current_desvar_values(self, sp_dvs:List[ScipyVariable]):   #doslovno
        x0 = []
        for spdv in sp_dvs:
            x0.append(spdv.value)
        return x0

    def _get_bounds_as_constraint_data(self,sp_dvs:List[ScipyVariable]):    #doslovno 
        consdata = []
        for spdv in sp_dvs:
            lbc = spdv.get_lb_COBYLA_constraint_data()
            if lbc is not None:
                consdata.append(lbc)
            ubc = spdv.get_ub_COBYLA_constraint_data()
            if ubc is not None:
                consdata.append(ubc)
        return consdata

    def _get_constraint_data(self,sp_cons:List[ScipyConstraint]):   #doslovno 3
        consdata = []
        for spcon in sp_cons:
            consdata.append(spcon.get_constraint_data_scipy())
        return consdata

    def optimize(self,desvars: List[DesignVariable], #ova se override-ana? metoda zove zapravo iz OptimizationProblem tipa objekta.. šalju se svi propertyji redom kao ovdje što se ocekuju, a zadnja dva su callback funkcije a to su tamo: self.evaluate i self.get_curren_sol
                 constraints: List[DesignConstraint],
                 objectives: List[DesignObjective],
                 x0:np.ndarray,
                 calback_evaluate, calback_get_current_solution) -> List[OptimizationProblemSolution]: #za to sto su callback_evaluate i callback_get_current_solution vidi optbase.py OptimizationProblem.optimize - to su dakle metode evaluate i get_current_sol u toj klasi
        
        (objfun,consdata,bnds) = self._generate_scipy_problem(desvars, constraints, objectives, calback_evaluate)  #tu se poziva generate_scipy_problem metoda kao arugment callback funkcije, to se poziva odmah! - ta funkcija naravno vraca (objfun, consdata, bnds) upravo tim redom 
                                                                                                                    #calback_evaluate i calback_get_current_solution su callback funkcije, i ne izvršavaju se odmah, vec se pozivaju kasnije(objfun,consdata,bnds) ce biti

        if self._method == 'SLSQP':
            sol = minimize(objfun, x0, constraints=consdata, method=self._method,options=self._options, bounds=bnds) #implementiran konacan poziv, x0 su pocetne vrijednosti design varijable consdata - treba biti lista rjecnika
        elif self._method == 'COBYLA':
            sol = minimize(objfun, x0, constraints=consdata, method=self._method,options=self._options)
        self._sol=sol                               #rjesenje se sprema u privatnu _sol varijablu, postoji property posto je privatna ._sol

        x = sol.x
        
        calback_evaluate(x)
        
        opt_sol = calback_get_current_solution()     #dihvaca posebno sadasnje rjesenje... ovaj callback zapravo
        solutions:List[OptimizationProblemSolution] = [] #valjda onaj history rjesenja? 
        solutions.append(opt_sol)
        return solutions                                    #lista rjesenja? 







