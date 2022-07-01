from abc import ABC,abstractmethod
from typing import List,Dict
from optbase import OptimizationProblem,OptimizationProblemSolution
from concurrent.futures import ProcessPoolExecutor,as_completed,Future
import numpy as np
from datetime import datetime
from os.path import dirname

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

    def optimize_task(self,op:OptimizationProblem,outfolder:str):
        op.optimize_and_write(outfolder)
        return op.full_name + ' - optimization finished - runtime: {} seconds\n'.format(op.opt_output.runtime)

    def execute(self):
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string+' - Job started')
        with ProcessPoolExecutor(max_workers= self._max_workers) as executor:
            outfolder='D:\\Development\moobench\\out'
            outfolder=dirname(dirname(__file__))+'\\out'
            futures:List[Future] = []
            for op in self._ops:
                futures.append(executor.submit(self.optimize_task,op,outfolder))
            for future in as_completed(futures):
                print(future.result())
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + ' - Job ended')

