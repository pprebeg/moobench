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


if __name__ == '__main__': #ako smo pokrenuli baš ovaj fajl, a ne ga pozvali izvana importiranjem.. onda je __name__ == '__main__' true. Na neki nacin nam ovo govori - ovo je program koji bi trebalo pokrenuti! 

    class EX_16_4_AnayisisModel(SimpleInputOutputArrayAnalysisExecutor): #takav analysis Executor je za jednostavne probleme - di zapišeš upravo na ovaj nacin funkcije cilja.. ulaz u izlaz se direktno izracunava.. 
        def __init__(self):
            super().__init__(2,4)

        def analyize(self): #u ovaj analyize (treba pravopisno stvari promijeniti) definiramo izgleda funkcije cilja, inarray - ulazni niz, outarray, izlazni niz, funckije cilja i vrijednosti ogranicenja
            self.outarray[0] = (self.inarray[0] - 1)**2.0 + (self.inarray[1] - 2.5)**2.0
            self.outarray[1] =  self.inarray[0] - 2 * self.inarray[1]   #nesto kao x1-2*x2>0
            self.outarray[2] = -self.inarray[0] - 2 * self.inarray[1]   #-x1-2*x2>0
            self.outarray[3] = -self.inarray[0] + 2 * self.inarray[1]   #-x1+2*x2
            return AnalysisResultType.OK

    am= EX_16_4_AnayisisModel() #am je analysis model! Pokazivac na objekt modela kreiranog ovdje! 
    op = OptimizationProblem()
    op.add_design_varible(DesignVariable('x1',NdArrayGetSetConnector(am.inarray,0),0.0, None))      #dodavanje dizajnerske varijable putem metode add_design_variable (lapsus calami), instanciranje DesignVariable, u kreiranu DesignVariable sprema novokreirani objekt konektora
    op.add_design_varible(DesignVariable('x2',NdArrayGetSetConnector(am.inarray, 1), 0.0, None))    #govori zapravo da je varijabla x1 ili x2, spremljena u am.inarrayu na mjestu 0 ili 1... Olakšava nama razumijevanje kaj je kaj.. 
    op.add_objective(DesignObjective('obj',NdArrayGetConnector(am.outarray,0)))                     #jedna je funkcija cilja.. VAZNO.. u argumentu odmah poziva instanciranje objekta.. bilo DesignVariable, bilo nekog drugog, pa ga pomocu ovih add metoda dodaje u listu vlastitih propertyja
    op.add_constraint(DesignConstraint('g1',NdArrayGetConnector(am.outarray,1), -2.0))              #tri su ogranicenja, pogledati kak se definira jesu li 
    op.add_constraint(DesignConstraint('g2',NdArrayGetConnector(am.outarray, 2), -6.0))             #tri argumenta - prvi je ime, drugi je konektor, treæi je vrijednost ogranièenja s desne strane
    op.add_constraint(DesignConstraint('g3',NdArrayGetConnector(am.outarray, 3), -2.0))
    op.add_analysis_executor(am)    #analysis executor je zaravo model, gdje su zapisane funkcije cilja, koje se sa metodom analyize izracunavaju.. To je prilicsno VAZNO za kompatibilnost svega. 
    opt_ctrl = {'method':'COBYLA'}  #ovo je dictionary koji se šalje konstruktoru ScipyOptimizationAlgorithm-a.. to znaci da su ostale postavke defaultne.. 
    #opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)         #postavljanje propertyja opt_algorithm (set metoda) i pritom spremanje objekta klase ScipyOptimizationAlgorithm
    res = op.optimize([2,0])    #pozivanje optimize! ovo [2,0] su pocetni x0.. op je tipa OptimizationProblem() iz optbase.py 
    print(res)                  #printanje rezultata - solutions iz optlib_scipyja se prenosi u optbase što završava ovdje 
    print(op.get_info())        #dobivanje informacija
    pass

    print('testing usage of optimization problem')
