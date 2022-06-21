import Analiza_okvira_v0_33 as ao
import numpy as np
from abc import ABC,abstractmethod
from pymoo.factory  import get_mutation

if __name__ == '__main__':
    try:
        from optbase import *
        from optlib_pymoo_proto import PymooOptimizationAlgorithm
    except ImportError:
        pass
else:
    try:
        from femdir.optbase import *
        from femdir.optlib_pymoo_proto import PymooOptimizationAlgorithm
    except ImportError:
        pass

if __name__ == '__main__':

    class Analiza_okvira_model(AnalysisExecutor): #potrebno napraviti neki drugaciji AnalysisExecutor mozda?
        
        def __init__(self):
            super().__init__() #ovo tu treba iz Analiza okvira možda pomoću posebnih funkcija dohvatiti.. Npr. get_number_of_design_variables, get_number_o itd. ako bude potrebno uopće za taj Analysis Executor. 

        def analyize(self): #u ovaj analyize (treba pravopisno stvari promijeniti) definiramo izgleda funkcije cilja, inarray - ulazni niz, outarray, izlazni niz, funckije cilja i vrijednosti ogranicenja

            ao.calculate_problem(model)  #PRORACUN u Analizi_okvira
            print(model.get_mass())
            
##            ao.get_stress_cons_lt_0()   #isto tako vrijedi i za Ratio tip constrainta. No, onda bi bilo smisleno i ostale na isti nacin implementirati! Npr. u obliku Ratia mogu biti i naprezanja! Obratiti pozornost na normiranje, istu skalu
                    
            return AnalysisResultType.OK



    
    InputFile=ao.Input_file('ponton.txt',0.2)
    InputFile.load_file()
    model = InputFile.create_model()            # u modelu je spremljen structure_obj preko kojeg imamo pristup svim funkcijama
    
    ao.calculate_problem(model)
    
    op = OptimizationProblem()
    am=Analiza_okvira_model()
    op.add_analysis_executor(am)
    
    #DIZAJNERSKE VARIJABLE

    #Primjer dodavanja svih parametara u dizajnerske varijable - ovaj interfejs dodavanja treba olakšati! Primjerice sa funkcijom- GetSection(ID) pa se pošalje ID. ili GetSectionParameter(pa se posalje ID. Ili funkcija GetSections
    #pa get parameters. uglavnom - nekak ovaj interface prema korisniku znatno pojednostavit.. 

##    sections_to_opt=[1]
##    for section_ID in sections_to_opt:
##        parameters:List = model.GetSectionParameters(section_ID)
##        for parameter in parameters:
##            op.add_design_variable(DesignVariable('x'+str(i),NdArrayGetSetConnector(parameters,index), section.bounds[index][0],section.bounds[index][1])

    sections_to_opt=[1,2,3]                 #Lista presjeka za optimizaciju
    
    lbs=[500, 5, None,None, 50, 5]
    ubs=[1500, 25, None, None, 500, 30] #None are just here to take place so that index works well

    print(len(op._desvars))
    
    for section in model.sections:
        
        if section.ID in sections_to_opt:

            if type(section)==ao.I_Beam:

                names=['hw','tw','bfa','tfa','bfb','tfb']
                index=0
            
                for (parameter,lb,ub,name) in zip(section.parameters, lbs, ubs, names):
                    
                    if ((index!=2) & (index!=3)):
                        
                        op.add_design_variable(DesignVariable(name+str(section.ID),NdArrayGetSetConnector(section.parameters,index), lb, ub)) #section.bounds[index][0],section.bounds[index][1]

                    index+=1

#DO OVDJE RADI!
                        
##    for section in model.sections:
##        
##        if section.ID in sections_to_opt:
##            
##            for parameter in section.parameters:
##                
##                index=section.parameters.index(parameter)
##                
##                if index!=3 and index!=4:
##                    op.add_design_variable(DesignVariable(name+str(i),NdArrayGetSetConnector(section.parameters,index),lb[index], ub[index] )) #section.bounds[index][0],section.bounds[index][1]
##                    i+=1

    #print(model.sections[0].parameters)                          
    #op._desvars[0].value=900
    #print(model.sections[0].parameters)   #optbase moze pristupiti i promijeniti parametre! 
                    
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
        print(ix)
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



                    #osmisliti dohvacanje ovih _desvars privatnih varijabli. -Na ovaj nacin se lako mogu kreirati ogranicenja - na ova dva nacina
                    #jos je potrebno ogranicenje naprezanja! - treba neki konektor da mozda poveze dvije stvari - naprezanje u gredi, a s desne strane je dozvoljeno naprezanje materijala. osmisliti kako to napraviti, a da korisniku
                    #ostane sto apstraktnije.. mozda nekako posebno pripremiti izlaz za ovaj program.. nekako pripremiti pokazivac na objekte greda i na objekte materijala... 

    mutation_obj=get_mutation('real_pm', eta=5, prob=0.2)
    alg_ctrl={'pop_size':10,'mutation':mutation_obj}       #u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl={'n_gen':20}                                                       #Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje 
    op.opt_algorithm = PymooOptimizationAlgorithm('ga', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)        #prvi argument string naziva algoritma, ostatak u obliku dictionary-ja ili tuple-a. mozda? Staviti da su defaultno None da se mogu ne poslati?
    #treba kreirati i termination criteria - pogledati je li potrebna nova klasa u optbase-u.

    #op.termination(n_gen=40)PymooTermination # Od termination criteria imamo sljedci izbor
                                   
    res = op.optimize([])    
    #print(res)                  
    #print(op.get_info())

    np.set_printoptions(suppress=True,precision=2)
    for beam in model.beams:
        print(beam.max_s)
    
    pass


