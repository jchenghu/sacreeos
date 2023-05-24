from typing import List

from sacreeos.err.commonerr import CommonErrors
from sacreeos.err.commonechecks import CommonChecks
from sacreeos.metric import MetricClass
from sacreeos.metric.metric import Metric
from sacreeos.metric.py_core.py_cider_scorer import PyCiderScorer, compute_doc_freq, cook_refss
from sacreeos.metric.c_core.invoker import Invoker
from sacreeos.utils.cli_args import Str2PositiveType


class CiderBase(Metric):
    """ Interface of the basic Cider metric."""

    DEFAULT_N = 4

    # Documentation / Messages for the manual generation
    QUESTIONS = ["n: int : maximum size of ngrams [default " + str(DEFAULT_N) + "]"]
    QUESTION_TEXT = "Insert the following arguments:\n\t" + \
                    '\n\t'.join(QUESTIONS)

    """ Construct the basic Cider metric interface.

    The CiderBase metric doesn't apply any form of penalty in the Cider computation.

    :param n: int : maximum size of ngrams
    :param corpus_refss: List[List[str]] : references for the corpus document frequencies
    """
    def __init__(self, n=DEFAULT_N, corpus_refss=None):

        super().__init__(MetricClass.CIDEr_Base, 'Cider', 'n'+str(n))
        self.precomp_corpus_ptr = None
        error_messages = self.check_n(n)
        if error_messages is not None:
            raise ValueError(error_messages)
        self.n = n
        self.corpus_df = None
        self.corpus_len = None
        if corpus_refss is not None:
            count_refss = [cook_refss(refs, n) for refs in corpus_refss]
            self.corpus_df = compute_doc_freq(count_refss)
            self.corpus_len = len(corpus_refss)

    def __del__(self):
        if self.precomp_corpus_ptr is not None:
            Invoker.free_cider_precomp_df(self.precomp_corpus_ptr)

    def compute(self, tests: List[str], refss: List[List[str]], fast_lib=False):
        """ Compute the basic Cider score.

        :param tests: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :param fast_lib: bool : if True, it makes use of the C implementation instead of Python
        :return: float, np.array(float) : the corpus cider score and an array of sentence-level cider scores
        """
        if not fast_lib:
            return PyCiderScorer.py_compute_score(MetricClass.CIDEr_Base, tests, refss, self.n,
                                                  self.corpus_df, self.corpus_len, penalty_args=None)
        else:
            cider_score, cider_array_scores, new_precomp_corpus_ptr = \
                Invoker.c_compute_score(self, tests, refss, self.corpus_df, self.corpus_len,
                                        self.precomp_corpus_ptr)
            self.precomp_corpus_ptr = new_precomp_corpus_ptr
            return cider_score, cider_array_scores

    def get_n(self):
        """ Return the maximum ngram size calculated from the inputs.
        :return: int : maximum size of ngrams
        """
        return self.n

    def check_n(self, n):
        CommonChecks.check_type("n", n, int)
        if n < 2:
            return CommonErrors.cant_be_lower_than("n", "2")
        return None

    # methods for manual generation
    @staticmethod
    def get_args_name_list_for_manual():
        return ['n']

    @staticmethod
    def get_args_default_list_for_manual():
        return [CiderBase.DEFAULT_N]

    @staticmethod
    def get_questions_text_for_manual():
        return CiderBase.QUESTIONS

    @staticmethod
    def get_args_data_convert_list_for_manual():
        return [lambda x: Str2PositiveType.to_digit(x)]






