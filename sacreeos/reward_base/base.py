from abc import abstractmethod

from sacreeos.err.commonerr import CommonErrors


class Base(object):
    """ Dummy based reward calculator. """

    DEFAULT_NSPI = 5

    """

    :param name_signature: str : set the base method signature
    :param args_signature: str : arguments to be included in the signature
    """
    def __init__(self, name_signature, args_signature, reward_class):
        self.name_signature = name_signature
        self.args_signature = args_signature
        self.reward_class = reward_class

    def get_signature(self):
        return self.name_signature + '[' + self.args_signature + ']'

    @abstractmethod
    def input_check(self, **args):
        CommonErrors.abstract_method_invokation_error()

    @abstractmethod
    def compute_based_reward(self, **args):
        CommonErrors.abstract_method_invokation_error()

    def get_class(self):
        return self.reward_class

