import Analiza_okvira_v0_33 as ao
import numpy as np

try:
    from optbase import *
    from optlib_scipy import ScipyOptimizationAlgorithm
except ImportError:
    pass


class Analiza_okvira_model(AnalysisExecutor):
    
    def __init__(self,model):
        super().__init__()
        self.model=model
        
    def analyze(self):

        #PRORACUN u Analizi_okvira
        ao.calculate_problem(self.model)  
                           
        return AnalysisResultType.OK

class Analiza_okvira_OptimizationProblem(OptimizationProblem):

    def __init__(self,name):

        super().__init__(name)
        
        InputFile=ao.Input_file('ponton_cijela_optimizacija.txt',40)
        InputFile.load_file()
        model = InputFile.create_model()                  
        ao.calculate_problem(model)

        #AnalysisExecutor
        am=Analiza_okvira_model(model)
        self.add_analysis_executor(am)
        
        #DIZAJNERSKE VARIJABLE

        sections_to_opt=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  
        
        lbs=[500, 5, None,None, 50, 5]
        ubs=[1500, 25, None, None, 500, 30]
        
        for section in model.sections:
            
            if section.ID in sections_to_opt:

                if type(section)==ao.I_Beam:

                    names=['hw','tw','bfa','tfa','bfb','tfb']
                    index=0
                
                    for (parameter,lb,ub,name) in zip(section.parameters, lbs, ubs, names):
                        
                        if ((index!=2) & (index!=3)):
                            
                            self.add_design_variable(DesignVariable(name+str(section.ID),NdArrayGetSetConnector(section.parameters,index), lb, ub)) 

                        index+=1
                        
        #FUNKCIJE CILJA
        m0 = model.get_mass()
        y0 = model.get_vertical_CG_position()
        print(f'm0 = {m0}')
        print(f'y0 = {y0}')
        self.add_objective(DesignObjective('mass',RatioGetCallbackConnector(model.get_mass, m0)))
        self.add_objective(DesignObjective('CG_y',RatioGetCallbackConnector(model.get_vertical_CG_position, y0)))

        #OGRANICENJA
        
        #NAPREZANJE
        i=0
        for beam in model.beams:
            self.add_constraint(DesignConstraint('stress'+str(i),CallbackGetConnector(beam.get_stress_over_limit), 1, ConstrType.LT))
            i+=1

        #DIMENZIJE

        dictionary={}   #dictionary of all desvars with their names
        
        for ix in range(self.num_var):
            
            desvar=self.get_desvar(ix)
            name=str(desvar.name)
            dictionary[name]=desvar

        i=0
        for section in model.sections:
            
            if section.ID in sections_to_opt:   #ovo definiranje u izdvojeni modul - kao minibiblioteku koja pruza pomoc u uobicajenim nacinima definiranja ogranicenja! Npr. posaljes ove nazive dictionaryje i ostalo i on to za te definira! 

                    if type(section)==ao.I_Beam:
                        
                        name1='bfb'+str(section.ID)
                        name2='tfb'+str(section.ID)
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),4))
                        i+=1
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),20, ConstrType.LT))
                        i+=1
                                          
                        name1='hw'+str(section.ID)
                        name2='tw'+str(section.ID)
##                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0))
##                        i+=1
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),90, ConstrType.LT))
                        i+=1

                        name1='bfb'+str(section.ID)
                        name2='hw'+str(section.ID)
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0.2))
                        i+=1
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0.5, ConstrType.LT))
                        i+=1

                        name1='tfb'+str(section.ID)
                        name2='tw'+str(section.ID)
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),1))
                        i+=1
                        self.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),3, ConstrType.LT))
                        i+=1


