

class Str2PositiveType(object):

    # WARNING: as the name suggest this class currently do not support inputs that can be negative,
    # it serves the purpose of converting metric input data so it should not need to any time soon...
    CONVERSION_FAILED = -1

    def __init__(self):
        pass

    @staticmethod
    def to_float(num):
        try:
            float_num = float(num)
            if float_num < 0:
                return Str2PositiveType.CONVERSION_FAILED
            return float_num
        except ValueError:
            return Str2PositiveType.CONVERSION_FAILED

    @staticmethod
    def to_digit(num):
        if num.isdigit():
            return int(num)
        else:
            return Str2PositiveType.CONVERSION_FAILED


