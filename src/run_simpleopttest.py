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


if __name__ == '__main__': #ako smo pokrenuli baš ovaj fajl, a ne ga pozvali izvana importiranjem.. onda je __name__ == '__main__' true

    class EX_16_4_AnayisisModel(SimpleInputOutputArrayAnalysisExecutor):
        def __init__(self):
            super().__init__(2,4)

        def analyize(self): #u ovaj analyize (treba pravopisno stvari promijeniti) definiramo izgleda funkcije cilja, inarray - ulazni niz, outarray, izlazni niz, funckije cilja i vrijednosti ogranicenja
            self.outarray[0] = (self.inarray[0] - 1)**2.0 + (self.inarray[1] - 2.5)**2.0
            self.outarray[1] =  self.inarray[0] - 2 * self.inarray[1]   #nesto kao x1-2*x2>0
            self.outarray[2] = -self.inarray[0] - 2 * self.inarray[1]   #-x1-2*x2>0
            self.outarray[3] = -self.inarray[0] + 2 * self.inarray[1]   #-x1+2*x2
            return AnalysisResultType.OK

    am= EX_16_4_AnayisisModel()
    op = OptimizationProblem()
    op.add_design_varible(DesignVariable('x1',NdArrayGetSetConnector(am.inarray,0),0.0, None))      #povezivanje pomocu konektora dizajnerske varijable
    op.add_design_varible(DesignVariable('x2',NdArrayGetSetConnector(am.inarray, 1), 0.0, None))
    op.add_objective(DesignObjective('obj',NdArrayGetConnector(am.outarray,0)))                     #jedna je funkcija cilja
    op.add_constraint(DesignConstraint('g1',NdArrayGetConnector(am.outarray,1), -2.0))              #tri su ogranicenja, pogledati kak se definira jesu li 
    op.add_constraint(DesignConstraint('g2',NdArrayGetConnector(am.outarray, 2), -6.0))
    op.add_constraint(DesignConstraint('g3',NdArrayGetConnector(am.outarray, 3), -2.0))
    op.add_analysis_executor(am)
    opt_ctrl = {'method':'COBYLA'}
    #opt_ctrl = {'method': 'SLSQP'}
    op.opt_algorithm = ScipyOptimizationAlgorithm(opt_ctrl)         #odabir algoritma optimizacije! pogledati optlib_scipy 
    res = op.optimize([2,0])    #pozivanje optimize!
    print(res)                  #printanje rezultata
    print(op.get_info())        #dobivanje informacija
    pass

    print('testing usage of optimization problem')
