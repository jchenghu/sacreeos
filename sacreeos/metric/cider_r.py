import os
from typing import List

from sacreeos.err.commonerr import CommonErrors
from sacreeos.err.commonechecks import CommonChecks
from sacreeos.metric import MetricClass
from sacreeos.metric.metric import Metric
from sacreeos.metric.py_core.py_cider_scorer import PyCiderScorer, compute_doc_freq, cook_refss
from sacreeos.metric.c_core.invoker import Invoker
from sacreeos.utils.cli_args import Str2PositiveType


class CiderR(Metric):
    """ Interface of the Cider-R metric. """

    DEFAULT_N = 4
    DEFAULT_REPEAT_COEFF = 0.8
    DEFAULT_LENGTH_COEFF = 0.2
    DEFAULT_ALPHA = 1.0

    # Documentation / Messages for the manual generation
    QUESTIONS = ["n: int : maximum size of ngrams [default " + str(DEFAULT_N) + "]",
                 "repeat_coeff: float : repeatition penalty weight in the geometric average of penalties [default "
                 + str(DEFAULT_REPEAT_COEFF) + "]",
                 "length_coeff: float : length penalty weight in the geometric average of penalties [default "
                 + str(DEFAULT_LENGTH_COEFF) + "]",
                 "alpha: float : length penalty deviance adjustment coefficient [default "
                 + str(DEFAULT_ALPHA) + "]"]
    QUESTION_TEXT = "Insert the following arguments:\n\t" + \
                    '\n\t'.join(QUESTIONS)

    """ Construct the Cider-R metric interface.

    Cider-R introduces the repeatition penalty.

    :param n: int : maximum size of ngrams calculated from the inputs
    :param repeat_coeff: float : repeatition penalty weight in the geometric average of penalties
    :param length_coeff: float : length penalty weight in the geometric average of penalties
    :param alpha: float : length penalty deviance adjustment coefficient
    :param corpus_refss: List[List[str]] : references from which the corpus document frequencies are calculated
    """
    def __init__(self, n=DEFAULT_N,
                 repeat_coeff=DEFAULT_REPEAT_COEFF,
                 length_coeff=DEFAULT_LENGTH_COEFF,
                 alpha=DEFAULT_ALPHA, corpus_refss=None):
        super().__init__(MetricClass.CIDEr_R, 'Cider-R',
                         'n'+str(n)+',rc' + str(repeat_coeff) +
                         ',lc' + str(length_coeff) + ',a' + str(alpha))
        self.precomp_corpus_ptr = None

        error_messages = self.check_n(n)
        if error_messages is not None:
            raise ValueError(error_messages)

        error_messages = self.check_length_and_repeat_coeffs(length_coeff, repeat_coeff)
        if error_messages is not None:
            raise ValueError(error_messages)

        error_messages = self.check_alpha(alpha)
        if error_messages is not None:
            raise ValueError(error_messages)

        self.n = n
        self.repeat_coeff = repeat_coeff
        self.length_coeff = length_coeff
        self.alpha = alpha
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
        """ Compute the Cider-R score.

        :param tests: list of sentences tested by the metric
        :param refss: list of list of sentences used as ground truths by the metric
        :param fast_lib: bool : if True, it makes use of the C implementation instead of Python
        :return: float, np.array(float) : the corpus cider-r score and an array of sentence-level cider-r scores
        """
        if not fast_lib:
            metrics_args = {'repeat_coeff': self.repeat_coeff, 'length_coeff': self.length_coeff,
                            'alpha': self.alpha}
            return PyCiderScorer.py_compute_score(MetricClass.CIDEr_R, tests, refss, self.n,
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

    def get_repeat_coeff(self):
        """ Get the repeatition penalty weight involved in the geometric average of penalties.
        :return: float :
        """
        return self.repeat_coeff
    
    def get_length_coeff(self):
        """ Get the length penalty weight involved in the geometric average of penalties.
        :return: float :
        """
        return self.length_coeff
    
    def get_alpha(self):
        """ Get the gaussian length penalty deviance adjustment coefficient.
        :return: float :
        """
        return self.alpha

    def check_n(self, n):
        CommonChecks.check_type("n", n, int)
        if n < 2:
            return CommonErrors.cant_be_lower_than("n", "2")
        return None

    # return True only if n is an accepted value
    def check_length_and_repeat_coeffs(self, length_coeff, repeat_coeff):
        CommonChecks.check_type("length_coeff", length_coeff, float)
        CommonChecks.check_type("repeat_coeff", repeat_coeff, float)
        if repeat_coeff <= 0.0 or length_coeff <= 0.0:
            return CommonErrors.cant_be_negative("All arguments")
        if repeat_coeff + length_coeff != 1.0:
            return CommonErrors.must_sum_up_to("Penalty coefficients "
                                               "repeat_coeff, length_coeff", "1")
        return None

    def check_alpha(self, alpha):
        CommonChecks.check_type("alpha", alpha, float)
        if alpha <= 0.0:
            return CommonErrors.cant_be_negative('alpha')
        return None

    @staticmethod
    def get_args_name_list_for_manual():
        return ['n', 'repeat_coeff', 'length_coeff', 'alpha']

    @staticmethod
    def get_args_default_list_for_manual():
        return [CiderR.DEFAULT_N, CiderR.DEFAULT_REPEAT_COEFF,
                CiderR.DEFAULT_LENGTH_COEFF, CiderR.DEFAULT_ALPHA]

    @staticmethod
    def get_questions_text_for_manual():
        return CiderR.QUESTIONS

    @staticmethod
    def get_args_data_convert_list_for_manual():
        return [lambda x: Str2PositiveType.to_digit(x), 
                lambda x: Str2PositiveType.to_float(x), 
                lambda x: Str2PositiveType.to_float(x), 
                lambda x: Str2PositiveType.to_float(x)]
