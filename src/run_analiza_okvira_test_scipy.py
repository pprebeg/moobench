import Analiza_okvira_v0_33 as ao
import numpy as np
from abc import ABC,abstractmethod
if __name__ == '__main__':
    try:
        from optbase import *
        from optlib_scipy import ScipyOptimizationAlgorithm
    except ImportError:
        pass
else:
    try:
        from femdir.optbase import *
        from femdir.optlib_scipy import ScipyOptimizationAlgorithm
    except ImportError:
        pass


if __name__ == '__main__': #__name__='__main__' samo ako smo pokrenuli ovaj file! Ako smo ga importirali, onda nije! 

    class Analiza_okvira_model(AnalysisExecutor):
        
        def __init__(self):
            super().__init__()

        def analyize(self):

            #PRORACUN u Analizi_okvira
            ao.calculate_problem(model)  
            #print(model.get_mass())
                               
            return AnalysisResultType.OK



    
    InputFile=ao.Input_file('ponton_cijela_optimizacija.txt',40)
    InputFile.load_file()
    model = InputFile.create_model()            # u modelu je spremljen structure_obj preko kojeg imamo pristup svim funkcijama i kreiranim objektima
    
    ao.calculate_problem(model)
    
    op = OptimizationProblem()
    am=Analiza_okvira_model()
    op.add_analysis_executor(am)
    
    #DIZAJNERSKE VARIJABLE

    #Primjer dodavanja svih parametara u dizajnerske varijable - ovaj interfejs dodavanja treba olakšati! Primjerice sa funkcijom- GetSection(ID) pa se pošalje ID. ili GetSectionParameter(pa se posalje ID. Ili funkcija GetSections
    #pa get parameters. uglavnom - nekak ovaj interface prema korisniku znatno pojednostavit.. 

    sections_to_opt=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]                 #Lista presjeka za optimizaciju
    
    lbs=[500, 5, None,None, 50, 5]
    ubs=[1500, 25, None, None, 500, 30] #None are just here to take place so that index works well
    
    for section in model.sections:
        
        if section.ID in sections_to_opt:

            if type(section)==ao.I_Beam:

                names=['hw','tw','bfa','tfa','bfb','tfb']
                index=0
            
                for (parameter,lb,ub,name) in zip(section.parameters, lbs, ubs, names):
                    
                    if ((index!=2) & (index!=3)):
                        
                        op.add_design_variable(DesignVariable(name+str(section.ID),NdArrayGetSetConnector(section.parameters,index), lb, ub)) #section.bounds[index][0],section.bounds[index][1]

                    index+=1
                    
    #FUNKCIJE CILJA

    
    op.add_objective(DesignObjective('mass',CallbackGetConnector(model.get_mass)))
                                       
    #U Analiza_okvira implementirana metoda Structure.get_mass()
    #Na slican nacin ugraditi jos koju metodu u model da korisniku bude lakse definirati neke ucestale funkcije cilja. N
    #Korisnik moze definirati i neku svoju funkciju cilja pristupajuci pojedinim podacima za pojedine Beam, Section i Property objekte - potrebno je poznavati program Analiza okvira!

    #OGRANICENJA
    

    
    #NAPREZANJE
    i=0
    for beam in model.beams:
        op.add_constraint(DesignConstraint('stress'+str(i),CallbackGetConnector(beam.get_stress_over_limit), 1, ConstrType.LT))
        i+=1

    #DIMENZIJE

    dictionary={}   #dictionary of all desvars with their names
    
    for ix in range(op.num_var):
        
        desvar=op.get_desvar(ix)
        name=str(desvar.name)
        dictionary[name]=desvar

    i=0
    for section in model.sections:
        if section.ID in sections_to_opt:   #ovo definiranje u izdvojeni modul - kao minibiblioteku koja pruza pomoc u uobicajenim nacinima definiranja ogranicenja! Npr. posaljes ove nazive dictionaryje i ostalo i on to za te definira! 

                if type(section)==ao.I_Beam:
                    
                    name1='bfb'+str(section.ID)
                    name2='tfb'+str(section.ID)
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),4))
                    i+=1
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),20, ConstrType.LT))
                    i+=1
                                      
                    name1='hw'+str(section.ID)
                    name2='tw'+str(section.ID)
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0))
                    i+=1
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),90, ConstrType.LT))
                    i+=1

                    name1='bfb'+str(section.ID)
                    name2='hw'+str(section.ID)
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0.2))
                    i+=1
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),0.5, ConstrType.LT))
                    i+=1

                    name1='tfb'+str(section.ID)
                    name2='tw'+str(section.ID)
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),1))
                    i+=1
                    op.add_constraint(DesignConstraint('g'+str(i),RatioDesignConnector(dictionary[name1],dictionary[name2]),3, ConstrType.LT))
                    i+=1

    x0=[]

    for i in range(op.num_var):
        x0.append(op.get_desvar(i).value)

        
    opt_ctrl = {'method':'COBYLA', 'maxiter':40000}  #ovo je dictionary koji se šalje konstruktoru ScipyOptimizationAlgorithm-a.. to znaci da su ostale postavke defaultne.. 
    #opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)         #postavljanje propertyja opt_algorithm (set metoda) i pritom spremanje objekta klase ScipyOptimizationAlgorithm
    res = op.optimize(x0)    #pozivanje optimize! ovo [2,0] su pocetni x0.. op je tipa OptimizationProblem() iz optbase.py 
    print(res)                  #printanje rezultata - solutions iz optlib_scipyja se prenosi u optbase što završava ovdje 
    print(op.get_info())        #dobivanje informacija
    pass

    print('testing usage of optimization problem')
