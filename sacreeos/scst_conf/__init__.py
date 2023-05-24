""" The implementation of requirements checks of the two possible SCST Configuration.

`scst_conf` package define a collection of input checks for both initialization and reward
 computation of the SCST algorithm. In particular, it ensures the special end of sequence (`eos`) token
 is included or omitted in all stages of the SCST algorithm according to one of the two selection:
 Standard and No<Eos> mode.

"""


# Classes interface for the maintainer, no the user
class ScstConfig:
    STANDARD_CONFIG = 1
    NOEOS_CONFIG = 2

    # This is a narrow-minded solution at the moment, currently don't expect to implement more
    # than just two classes
    SELECTION_YES = 'yes'
    SELECTION_NO = 'no'


# Documentation / Messages for the signature manual generation
SCSTINIT_HELP_TEXT = "If the CIDEr-based reward function computes the tf-idfs using the training corpus, " + \
                     "then the answer is 'yes'.\n" + \
                     "Otherwise, if the tf-idfs are calculated using only " + \
                     "the reference descriptions of each image, the answer is 'no'.\n" + \
                     "Extra: In the case of MS-COCO 2014 data set, it is suggested to initialize tf-idfs " + \
                     "using the training corpus, since the number of reference descriptions is small.\n"

SCSTMODE_HELP_TEXT = "If the End-of-Sequence token is included in the reward and eventually in the " + \
                     "initialization, then the answer is 'yes'.\n" + \
                     "If the End-of-Sequence token is excluded from the computation of rewards and " + \
                     "eventually in the initialization, then the answer is 'no'.\n" + \
                     "In case of CIDEr based reward, the End-of-Sequence token plays " + \
                     "an important role and affects both evaluation reward and descriptions.\n" + \
                     "If omitted, the reward in both training and evaluation will be higher, " + \
                     "but descriptions will suffer from artifacts such as trivial words termination.\n" + \
                     "Because of the importance of the matter, no default answer is provided."

SCSTINIT_POSSIBLE_ANS = {'y': ScstConfig.SELECTION_YES, 'n': ScstConfig.SELECTION_NO}
SCSTINIT_DEFAULT_ANS = 'y'
SCSTINIT_QUESTION_TEXT = "Are the tf-idfs initialized with the training corpus? [default: " + SCSTINIT_DEFAULT_ANS + "]"

SCSTMODE_QUESTION_TEXT = "Is the End-of-Sequence token included during both initialization and reward computation? "
SCSTMODE_POSSIBLE_ANS = {'y': ScstConfig.SELECTION_YES, 'n': ScstConfig.SELECTION_NO}
# SCST_DEFAULT_ANSWER_TEXT = ... < No default value is provided
SCSTMODE_DEFAULT_ANS = 'h'

