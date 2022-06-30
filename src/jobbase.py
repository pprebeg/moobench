from abc import ABC,abstractmethod
from typing import List,Dict
from optbase import OptimizationProblem,OptimizationProblemSolution
from concurrent.futures import ProcessPoolExecutor
import numpy as np

class Job:
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def execute(self):
        pass

class MultibjectiveOptimizationComparer(Job):
    def __init__(self, name,ops:List[OptimizationProblem],max_workers=2):
        super(MultibjectiveOptimizationComparer, self).__init__(name)
        self._ops:List[OptimizationProblem] = ops
        self._max_workers = max_workers



    def execute(self):
        with ProcessPoolExecutor(max_workers= self._max_workers) as executor:
            outfolder='D:\\Development\moobench\\out'
            for op in self._ops:
                executor.submit(op.optimize_and_write(outfolder))


