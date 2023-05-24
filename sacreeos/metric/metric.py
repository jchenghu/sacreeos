from typing import List
from abc import abstractmethod

from sacreeos.err.commonerr import CommonErrors


class Metric:
    """ Metric interface providing the basic functionalities.

    :param metric_class: int : set the metric class
    :param name_signature: str : set the metric signature
    :param args_signature: str : arguments to be included in the signature
    """
    def __init__(self,
                 metric_class,
                 name_signature,
                 args_signature):
        self.metric_class = metric_class
        self.signature = name_signature + '[' + args_signature + ']'

    @abstractmethod
    def compute(self, tests: List[str], refss: List[List[str]], fast_lib=False):
        CommonErrors.abstract_method_invokation_error()

    def input_check(self, test, refss):
        if not (isinstance(refss, list) and isinstance(refss[0], list) and isinstance(refss[0][0], str)):
            raise TypeError("`refss` must be list of list of strings.")
        if not (isinstance(test, list) and isinstance(test[0], list)):
            raise TypeError("`test` must be list of list of strings.")

    def get_metric_class(self):
        """ Get the metric class identifier.
        :return: int :
        """
        return self.metric_class

    def get_signature(self):
        """ Get the signature of the metric.
        :return: str :
        """
        return self.signature
