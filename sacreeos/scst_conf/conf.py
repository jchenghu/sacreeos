from abc import abstractmethod

from sacreeos.err.commonerr import CommonErrors


class ScstConf:
    """ Dummy SCST arguments controller.

    :param signature: str : signature assigned to the SCST configuration
    :param is_initialized: bool : set the SCST signature according to whether it required an initialization step or not
    """
    def __init__(self, signature, is_initialized, scst_config):
        if not isinstance(is_initialized, bool):
            raise TypeError("`is_initialized` must be boolean")
        self.signature = signature
        self.is_initialized = is_initialized
        self.initialization_signature = '_wInit' if self.is_initialized else '_w/oInit'
        self.scst_config = scst_config

    def get_signature(self):
        """ Get the signature of the metric.
        :return: str :
        """
        return self.signature + self.initialization_signature

    @abstractmethod
    def cider_init_check(self, **args):
        CommonErrors.abstract_method_invokation_error()

    @abstractmethod
    def compute_reward(self):
        CommonErrors.abstract_method_invokation_error()

    @abstractmethod
    def input_check(self, **args):
        CommonErrors.abstract_method_invokation_error()

    def get_class(self):
        return self.scst_config
