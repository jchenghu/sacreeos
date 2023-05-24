""" Common error outputs and input checks. """


class CommonErrors(object):

    prefix = 'SacreEOS: '

    @staticmethod
    def invalid_scst_class():
        return CommonErrors.prefix + "Invalid SCST configuration is provided."

    @staticmethod
    def invalid_metric_class():
        return CommonErrors.prefix + "Invalid Metric is provided."

    @staticmethod
    def invalid_reward_base_class():
        return CommonErrors.prefix + "Invalid Reward base was provided."

    @staticmethod
    def cant_be_negative(str):
        return CommonErrors.prefix + str + " must be non negative."

    @staticmethod
    def must_sum_up_to(str_input, str_sum):
        return CommonErrors.prefix + str_input + " must sum up to " + str_sum + "."

    @staticmethod
    def cant_be_lower_than(str_input, str_sum):
        return CommonErrors.prefix + str_input + " can't be lower than " + str_sum + "."
