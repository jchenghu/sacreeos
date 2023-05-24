from typing import List

from sacreeos.err.commonerr import CommonErrors
from sacreeos.err.commonechecks import CommonChecks
from sacreeos.metric import MetricClass
from sacreeos.metric.metric import Metric
from sacreeos.metric.py_core.py_cider_scorer import PyCiderScorer, compute_doc_freq, cook_refss
from sacreeos.metric.c_core.invoker import Invoker
from sacreeos.utils.cli_args import Str2PositiveType


class CiderD(Metric):
    """ Interface of the Cider-D metric. """

    DEFAULT_N = 4
    DEFAULT_SIGMA = 6.0

    # Documentation / Messages for the manual generation
    QUESTIONS = ['n: int : maximum size of ngrams [default ' + str(DEFAULT_N) + ']',
                 'sigma: float : length penalty gaussian deviance [default ' + str(DEFAULT_SIGMA) + ']']
    QUESTION_TEXT = "Insert the following arguments:\n\t" + \
                    '\n\t'.join(QUESTIONS)

    """ Construct the Cider-D metric interface.

    The Cider-D metric computes the Cider score but addresses length disparities between input sequences
    and references introducing a length penalty.

    :param n: int : maximum size of ngrams calculated from the inputs
    :param sigma: float : length penalty gaussian deviance
    :param corpus_refss: List[List[str]] : references from which the corpus document frequencies are calculated
    """
    def __init__(self, n=DEFAULT_N, sigma=DEFAULT_SIGMA, corpus_refss=None):
        super().__init__(MetricClass.CIDEr_D, 'Cider-D', 'n'+str(n)+','+'s'+str(sigma))
        self.precomp_corpus_ptr = None
        error_messages = self.check_n(n)
        if error_messages is not None:
            raise ValueError(error_messages)
        error_messages = self.check_sigma(sigma)
        if error_messages is not None:
            raise ValueError(error_messages)

        self.n = n
        self.sigma = sigma
        self.corpus_df = None
        self.corpus_len = None
        if corpus_refss is not None:
            count_refss = [cook_refss(refs) for refs in corpus_refss]
            self.corpus_df = compute_doc_freq(count_refss)
            self.corpus_len = len(corpus_refss)

    def __del__(self):
        if self.precomp_corpus_ptr is not None:
            Invoker.free_cider_precomp_df(self.precomp_corpus_ptr)

    def compute(self, tests: List[str], refss: List[List[str]], fast_lib=False):
        """ Compute the Cider-D score.

        :param test: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :param fast_lib: bool : if True, it makes use of the C implementation instead of Python
        :return: float, np.array(float) : the corpus cider-d score and an array of sentence-level cider-d scores
        """
        if not fast_lib:
            metrics_args = {'sigma': self.sigma}
            return PyCiderScorer.py_compute_score(MetricClass.CIDEr_D, tests, refss, self.n,
                                                  self.corpus_df, self.corpus_len, metrics_args)
        else:
            cider_score, cider_array_scores, new_precomp_corpus_ptr = \
                Invoker.c_compute_score(self, tests, refss, self.corpus_df, self.corpus_len,
                                        self.precomp_corpus_ptr)
            self.precomp_corpus_ptr = new_precomp_corpus_ptr
            return cider_score, cider_array_scores

    def get_n(self):
        """ Return the maximum ngram size calculated from the inputs.
        :return: int :
        """
        return self.n

    def get_sigma(self):
        """ Return the gaussial length penalty deviance.
        :return: float :
        """
        return self.sigma

    def check_n(self, n):
        CommonChecks.check_type("n", n, int)
        if n < 2:
            return CommonErrors.cant_be_lower_than("n", "2")
        return None

    def check_sigma(self, sigma):
        CommonChecks.check_type("sigma", sigma, float)
        if sigma <= 0.0:
            return CommonErrors.cant_be_negative('sigma')
        return None

    @staticmethod
    def get_args_name_list_for_manual():
        return ['n', 'sigma']

    @staticmethod
    def get_args_default_list_for_manual():
        return [CiderD.DEFAULT_N, CiderD.DEFAULT_SIGMA]

    @staticmethod
    def get_questions_text_for_manual():
        return CiderD.QUESTIONS

    @staticmethod
    def get_args_data_convert_list_for_manual():
        return [lambda x: Str2PositiveType.to_digit(x), lambda x: Str2PositiveType.to_float(x)]

