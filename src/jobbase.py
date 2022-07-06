from abc import ABC,abstractmethod
from typing import List,Dict
from optbase import OptimizationProblem,OptimizationProblemMultipleSolutions
from concurrent.futures import ProcessPoolExecutor,as_completed,Future
import numpy as np
from datetime import datetime
from os.path import dirname
import os

class Job:
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def execute(self):
        pass

class MultibjectiveOptimizationComparer(Job):
    def __init__(self, name:str,max_workers:int,out_folder_path:str,
                 ops:List[OptimizationProblem],
                 ref_pareto_front:OptimizationProblemMultipleSolutions):
        super().__init__(name)
        self._max_workers = max_workers
        self._out_folder_path = out_folder_path
        self._ops:List[OptimizationProblem] = ops
        self._ref_pareto_front = ref_pareto_front

    def _optimize_task(self, op:OptimizationProblem, outfolder:str):
        successful = op.optimize_and_write(outfolder)
        if successful:
            return 'Optimization finished:\n'+ op.opt_output.get_info()
        else:
            return 'Optimization finished:\n Algorithm did not converge!'

    def execute(self):
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string+' - Job started')
        isExist = os.path.exists(self._out_folder_path)
        if not isExist:
            os.makedirs(self._out_folder_path)
        with ProcessPoolExecutor(max_workers= self._max_workers) as executor:
            futures:List[Future] = []
            for op in self._ops:
                futures.append(executor.submit(self._optimize_task, op, self._out_folder_path))
            for future in as_completed(futures):
                print(future.result())
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + ' - Job ended')

class MultibjectiveOptimizationComparerFromWrittenResults(MultibjectiveOptimizationComparer):
    def __init__(self, name:str,max_workers:int,out_folder_path:str,
                 ops:List[OptimizationProblem],
                 ref_pareto_front:OptimizationProblemMultipleSolutions):
        super().__init__(name,max_workers,out_folder_path,ops,ref_pareto_front)

    def _readresults_or_optimize_task(self, op:OptimizationProblem, outfolder:str):
        op.read_results_or_optimize_and_write(outfolder,None,self._ref_pareto_front)
        return 'Read results or optimization finished:\n'+ op.opt_output.get_info()

    def execute(self):
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string+' - Job started')
        isExist = os.path.exists(self._out_folder_path)
        if not isExist:
            os.makedirs(self._out_folder_path)
        with ProcessPoolExecutor(max_workers= self._max_workers) as executor:
            futures:List[Future] = []
            for op in self._ops:
                futures.append(executor.submit(self._readresults_or_optimize_task, op, self._out_folder_path))
            for future in as_completed(futures):
                print(future.result())
            #
        dt_string = (datetime.now()).strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string + ' - Job ended')

