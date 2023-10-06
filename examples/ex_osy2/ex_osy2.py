try:
    from moobench.optbase import *
    from moobench.optlib_scipy import ScipyOptimizationAlgorithm
except ImportError:
    print('Error: moobench library not installed or not found!')
    pass


class OSY2_AnMod(SimpleInputOutputArrayAnalysisExecutor):
    # 6 varijabli, 6 ogranicenja, 2 cilja, 

    def __init__(self):
        super().__init__(6,8)

    def analyze(self):
        x1 = self.inarray[0]
        x2 = self.inarray[1]
        x3 = self.inarray[2]
        x4 = self.inarray[3]
        x5 = self.inarray[4]
        x6 = self.inarray[5]

        # objectives
        self.outarray[0] = -(25 * (x1 - 2) ** 2 + (x2 - 2) ** 2 + (x3 - 1) ** 2 + (x4 - 4) ** 2 + (x5 - 1) ** 2)
        self.outarray[1] = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2

        # constraints
        self.outarray[2] = (2 - x1 - x2) * (-1)
        self.outarray[3] = (x1 + x2 - 6) * (-1)
        self.outarray[4] = (x2 - x1 - 2) * (-1)
        self.outarray[5] = (x1 - 3 * x2 - 2) * (-1)
        self.outarray[6] = ((x3 - 3) ** 2 + x4 - 4) * (-1)
        self.outarray[7] = 4 - (x5 - 3) ** 2 - x6

        return AnalysisResultType.OK

class OSY2_OptProb(OptimizationProblem):
    def __init__(self,name=''):
        if name == '':
            name = 'Osyczka2'
        super().__init__(name)
        am = OSY2_AnMod()
        # variables
        self.add_design_variable(DesignVariable('x1', NdArrayGetSetConnector(am.inarray, 0), 0.0, 10.0))
        self.add_design_variable(DesignVariable('x2', NdArrayGetSetConnector(am.inarray, 1), 0.0, 10.0))
        self.add_design_variable(DesignVariable('x3', NdArrayGetSetConnector(am.inarray, 2), 1.0, 5.0))
        self.add_design_variable(DesignVariable('x4', NdArrayGetSetConnector(am.inarray, 3), 0.0, 6.0))
        self.add_design_variable(DesignVariable('x5', NdArrayGetSetConnector(am.inarray, 4), 1.0, 5.0))
        self.add_design_variable(DesignVariable('x6', NdArrayGetSetConnector(am.inarray, 5), 0.0, 10.0))
        # objectives
        self.add_objective(DesignObjective('obj1', NdArrayGetConnector(am.outarray, 0)))
        self.add_objective(DesignObjective('obj2', NdArrayGetConnector(am.outarray, 1)))
        # constraints
        self.add_constraint(DesignConstraint('g1', NdArrayGetConnector(am.outarray, 2), 0.0))
        self.add_constraint(DesignConstraint('g2', NdArrayGetConnector(am.outarray, 3), 0.0))
        self.add_constraint(DesignConstraint('g3', NdArrayGetConnector(am.outarray, 4), 0.0))
        self.add_constraint(DesignConstraint('g4', NdArrayGetConnector(am.outarray, 5), 0.0))
        self.add_constraint(DesignConstraint('g5', NdArrayGetConnector(am.outarray, 6), 0.0))
        self.add_constraint(DesignConstraint('g6', NdArrayGetConnector(am.outarray, 7), 0.0))
        self.add_analysis_executor(am)