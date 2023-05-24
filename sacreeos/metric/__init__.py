""" The implementation of metrics used as reward functions.

`Metric` can be divided in two the `core` packages and `interfaces`:
    1) The `*_core` is where the heart of the package lies, it contains all the algorithms
       in both C (`c_core`) and Python (`py_core`) implementations.

    2) `interfaces` consists of classes which main goal consist of storing the metrics default values,
       the input arguments or related objects. It also performs initializations or whatever operation may be
       required prior to the actual score computation.
       And as the name suggests, it hides the underlying C/Py core methods calls and request just
       the essential arguments from the user.

"""

from sacreeos import __REPO_LINK__


# Classes interface for the maintainer, no the user
class MetricClass:
    CIDEr_Base = 1
    CIDEr_D = 2
    CIDEr_R = 3
    BLEU = 4

    @staticmethod
    def answer_to_class(ans):
        if ans == 'd':
            return MetricClass.CIDEr_D
        elif ans == 'r':
            return MetricClass.CIDEr_R
        elif ans == 'c':
            return MetricClass.CIDEr_Base
        elif ans == 'b':
            return MetricClass.BLEU
        else:
            raise ValueError("Answer not supported.")

    CIDEr_Base_STRING_ID = 'CIDEr_Base'
    CIDEr_D_STRING_ID = 'CIDEr-D'
    CIDEr_R_STRING_ID = 'CIDEr-R'
    BLEU_STRING_ID = 'Bleu'


# Documentation / Messages for the manual generation
METRIC_HELP_TEXT = "The reward refers to the metric used in the evaluation of the sentences and  " + \
                   "the function maximized by SCST\n" + \
                   "CIDEr/CIDEr-D reference:\n" + \
                   "\tTitle: \"CIDEr: Consensus-based Image Description Evaluation\"\n" + \
                   "\tAuthors: Ramakrishna Vedantam, C. Lawrence Zitnick and Devi Parikh.\n" + \
                   "\tLink: https://arxiv.org/abs/1411.5726\n" + \
                   "CIDEr-R reference:\n" + \
                   "\tTitle: \"CIDEr-R: Robust Consensus-based Image Description Evaluation\"\n" + \
                   "\tAuthors: Gabriel Oliveira dos Santos, Esther Luna Colombini and Sandra Avila.\n" + \
                   "\tLink: https://arxiv.org/abs/2109.13701\n" + \
                   "BLEU reference:\n" + \
                   "\tTitle: \"Bleu: a Method for Automatic Evaluation of Machine Translation\"\n" + \
                   "\tAuthors: Kishore Papineni, Salim Roukos, Todd Ward and Wei-Jing Zhu.\n" + \
                   "\tLink: https://dl.acm.org/doi/10.3115/1073083.1073135\n" + \
                   "If the reward function you are looking for is not enlisted here, \n" + \
                   "you may want to create an issue: " + __REPO_LINK__

METRIC_POSSIBLE_ANS = {'d': MetricClass.CIDEr_D_STRING_ID, 'r': MetricClass.CIDEr_R_STRING_ID,
                       'c': MetricClass.CIDEr_Base_STRING_ID, 'b': MetricClass.BLEU_STRING_ID}
METRIC_DEFAULT_ANS = 'd'
METRIC_QUESTION_TEXT = "What is the reward metric ? [default: " + METRIC_DEFAULT_ANS + "]"


