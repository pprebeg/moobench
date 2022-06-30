import numpy as np
from pymoo.factory  import get_mutation, get_crossover, get_selection
from pymoo.factory import get_visualization



if __name__ == '__main__':
    try:
        from optbase import *
        from optlib_pymoo_proto import PymooOptimizationAlgorithmSingle, PymooOptimizationAlgorithmMulti
    except ImportError:
        pass
else:
    try:
        from femdir.optbase import *
        from femdir.optlib_pymoo_proto import PymooOptimizationAlgorithmSingle, PymooOptimizationAlgorithmMulti
    except ImportError:
        pass

if __name__ == '__main__':

##    # simple binary tournament for a single-objective algorithm
##    def binary_tournament(pop, P, algorithm, **kwargs):
##
##        # The P input defines the tournaments and competitors
##        n_tournaments, n_competitors = P.shape
##
##        if n_competitors != 2:
##            raise Exception("Only pressure=2 allowed for binary tournament!")
##
##        # the result this function returns
##        import numpy as np
##        S = np.full(n_tournaments, -1, dtype=np.int)
##
##        # now do all the tournaments
##        for i in range(n_tournaments):
##            a, b = P[i]
##
##            # if the first individiual is better, choose it
##            if pop[a].F < pop[a].F:
##                S[i] = a
##
##            # otherwise take the other individual
##            else:
##                S[i] = b
##
##        return S

    class Osyczka2(SimpleInputOutputArrayAnalysisExecutor): #12 boundsa, 18-12=6 ogranicenja, 2 cilja, 6 dizajnerskih parametara

        def __init__(self):
            super().__init__(6,8) #ovo tu treba iz Analiza okvira možda pomoću posebnih funkcija dohvatiti.. Npr. get_number_of_design_variables, get_number_o itd. ako bude potrebno uopće za taj Analysis Executor.

        def analyze(self): #u ovaj analyize (treba pravopisno stvari promijeniti) definiramo izgleda funkcije cilja, inarray - ulazni niz, outarray, izlazni niz, funckije cilja i vrijednosti ogranicenja

            #DIZAJNERSKE VARIJABLE
            x1 = self.inarray[0]
            x2 = self.inarray[1]
            x3 = self.inarray[2]
            x4 = self.inarray[3]
            x5 = self.inarray[4]
            x6 = self.inarray[5]

            #FUNKCIJE CILJA
            self.outarray[0] = -(25*(x1-2)**2+(x2-2)**2+(x3-1)**2+(x4-4)**2+(x5-1)**2)
            self.outarray[1] = x1**2+x2**2+x3**2+x4**2+x5**2+x6**2

            #FUNKCIJE OGRANICENJA
            self.outarray[2] = (2-x1-x2)*(-1)
            self.outarray[3] = (x1+x2-6)*(-1)
            self.outarray[4] = (x2-x1-2)*(-1)
            self.outarray[5] = (x1-3*x2-2)*(-1)
            self.outarray[6] = ((x3-3)**2+x4-4)*(-1)
            self.outarray[7] = 4-(x5-3)**2-x6

            return AnalysisResultType.OK

    bnds = [[0,10],[0,10],[1,5],[0,6],[1,5],[0,10]]

    op = OptimizationProblem()
    am=Osyczka2()
    op.add_analysis_executor(am)

    #DIZAJNERSKE VARIJABLE

    op.add_design_variable(DesignVariable('x1',NdArrayGetSetConnector(am.inarray,0), 0.0, 10.0))
    op.add_design_variable(DesignVariable('x2',NdArrayGetSetConnector(am.inarray, 1), 0.0, 10.0))
    op.add_design_variable(DesignVariable('x3',NdArrayGetSetConnector(am.inarray,2), 1.0, 5.0))
    op.add_design_variable(DesignVariable('x4',NdArrayGetSetConnector(am.inarray, 3), 0.0, 6.0))
    op.add_design_variable(DesignVariable('x5',NdArrayGetSetConnector(am.inarray,4), 1.0, 5.0))
    op.add_design_variable(DesignVariable('x6',NdArrayGetSetConnector(am.inarray, 5), 0.0, 10.0))

    #FUNKCIJE CILJA
    op.add_objective(DesignObjective('obj1',NdArrayGetConnector(am.outarray,0)))
    op.add_objective(DesignObjective('obj2',NdArrayGetConnector(am.outarray,1)))

    #FUNKCIJE OGRANICENJA
    #VEC SU PRIPREMLJENA U OBLIKU >=0, DESNA STRANA JE PREBACENA LIJEVO JE DESNO 0
    op.add_constraint(DesignConstraint('g1',NdArrayGetConnector(am.outarray,2), 0.0))
    op.add_constraint(DesignConstraint('g2',NdArrayGetConnector(am.outarray, 3), 0.0))
    op.add_constraint(DesignConstraint('g3',NdArrayGetConnector(am.outarray, 4), 0.0))
    op.add_constraint(DesignConstraint('g4',NdArrayGetConnector(am.outarray,5), 0.0))
    op.add_constraint(DesignConstraint('g5',NdArrayGetConnector(am.outarray, 6), 0.0))
    op.add_constraint(DesignConstraint('g6',NdArrayGetConnector(am.outarray, 7), 0.0))
    
    # dio postavki specifican za pymoo

    pop_size = 100
    num_iter = 100
    max_evaluations = pop_size*num_iter

    mutation_obj = get_mutation('real_pm', eta=20, prob=1.0/6)    # Check
    crossover_obj = get_crossover('real_sbx', eta=20, prob=1.0)     #Check
##    selection_obj = get_selection('tournament',{'func_comp' : binary_tournament}) Selection je random Selection je odabir za rekombinaciju 
    
    alg_ctrl={'pop_size':pop_size,'mutation':mutation_obj, 'crossover':crossover_obj}       #u obliku dictionary-ja se salju svi keyword argumenti! Dodatni argumenti poput tuple-a('n_gen',40) - al to su kriteriji izgleda termination
    term_ctrl={'n_eval':max_evaluations}                                                       #Ovo treba biti u obliku liste. Primjer je dan kako se u obliku liste šalje
    op.opt_algorithm = PymooOptimizationAlgorithmMulti('nsga2', alg_ctrl=alg_ctrl, term_ctrl=term_ctrl)        #prvi argument string naziva algoritma, ostatak u obliku dictionary-ja ili tuple-a.
    #op.termination(n_gen=40)PymooTermination # Od termination criteria imamo sljedci izbor

    res = op.optimize([])
    #print(res)
    print(op.get_info())
    
    pairs = []
    for pair in op._solutions:
        pairs.append(pair.objs.tolist())
    print(pairs)
    pairs = np.array(pairs)
    plot = get_visualization("scatter")
    plot.add(pairs, color="green", marker="x")
    #plot.grid()
    plot.show()
