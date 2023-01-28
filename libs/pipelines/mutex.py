import random
from typing import List
from mmaction.datasets import PIPELINES
from mmaction.datasets.pipelines import Compose


@PIPELINES.register_module()
class MutexPipelines:
    """
    This class is a container for several pipeline
    """
    def __init__(self, mutex_pipelines: List, probs: List):
        if len(probs) != len(mutex_pipelines):
            raise ValueError("the len(probs) must equal len(mutex_pipelines)")

        self.mutex_pipelines = []
        for pipeline_config in mutex_pipelines:
            self.mutex_pipelines.append(Compose(pipeline_config))
        self.probs = probs

    def __call__(self, results):
        for pipeline, prob in zip(self.mutex_pipelines, self.probs):
            if random.random() < prob:
                return pipeline(results)
        return results


@PIPELINES.register_module()
class PrintPipelines:
    """
    The purpose of this class is to help debug
    """
    def __init__(self, message):
        self.message = message

    def __call__(self, result):
        print(self.message)
        return result
