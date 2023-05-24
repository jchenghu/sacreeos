""" The implementation of based rewards calculators. """

from sacreeos.reward_base.base import Base


# Classes interface for the maintainer, no the user
class BaseRewardClass:
    AVERAGE = 1
    GREEDY = 2

    # not exactly a name, but we denote it as NAME for it has the same role of the
    GREEDY_BASE_STRING_ID = 'greedy decoding'
    AVERAGE_BASE_STRING_ID = 'mean over the number of samples'

# Documentation / Messages for the signature manual generation
BASE_HELP_TEXT = "Select 'g' in case the base sequences are made with the highest probability words " + \
                 "(the decoder act 'greedily').\n" + \
                 "Select 'm' in case the base is simply the mean over the number of sampled descriptions.\n"
BASE_ARGS_HELP_TEXT = "For each image several descriptions can be sampled. On the COCO data set, the number " + \
                      "is typically 5 (which has nothing to do with the number of references). Moreover, " + \
                      "it must be > 1 in case of 'mean' type base.\n"

BASE_POSSIBLE_ANS = {'g': BaseRewardClass.GREEDY_BASE_STRING_ID, 'm': BaseRewardClass.AVERAGE_BASE_STRING_ID}
BASE_DEFAULT_ANS = 'g'
BASE_QUESTION_TEXT = "How is the base calculated? [default: " + BASE_DEFAULT_ANS + "]"


# Little hack on the "(number)" indication: see #PTR_KQ"8
BASE_ARGS_DEFAULT_ANS = Base.DEFAULT_NSPI
BASE_ARGS_QUESTION_TEXT = "How many descriptions are sampled for each image? (number) " +\
                           "[default: " + str(Base.DEFAULT_NSPI) + "] "
