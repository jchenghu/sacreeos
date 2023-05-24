import os
from typing import List

from sacreeos.metric import MetricClass
from sacreeos.metric.metric import Metric
from sacreeos.metric.py_core.py_bleu_scorer import PyBleuScorer
from sacreeos.metric.c_core.invoker import Invoker


class BLEU(Metric):
    """ Interface of the BLEU metric.

    Computes the corpus level and sentence level BLEU scores for n=4.
    """

    def __init__(self):
        super().__init__(MetricClass.BLEU, 'BLEU', 'n4')
        # TO-DO: if required: extends BLEU n to > 4

    def compute(self, tests: List[str], refss: List[List[str]], fast_lib=False):
        """ Compute the BLEU score.

        :param tests: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :param fast_lib: bool : if True, it make use of the C implementation instead of Python
        :return: Sequence[float], Sequence[Sequence[float]] : the corpus bleu scores and an array of sentence-level bleu scores
        """
        if not fast_lib:
            return PyBleuScorer.py_compute_score(tests, refss)
        else:
            return Invoker.c_compute_score(self, tests, refss, None, None,
                                           'pid' + str(os.getpid()) + '-obj_id-' + str(id(self)))
