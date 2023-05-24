
class CommonChecks:

    @staticmethod
    def abstract_method_invokation_error():
        raise NotImplementedError("This method should be overriden by child classes.")

    @staticmethod
    def check_list_list_sentences_format(input_name, input_obj):
        if (input_obj is None) \
                or (not (isinstance(input_obj, list) and isinstance(input_obj[0], list)
                         and isinstance(input_obj[0][0], str))):
            raise TypeError("`" + input_name + "` is required in List[List[str]] format.")

    @staticmethod
    def check_list_sentences_format(input_name, input_obj):
        if (input_obj is None) \
                or (not (isinstance(input_obj, list) and isinstance(input_obj[0], str))):
            raise TypeError("`" + input_name + "` is required in List[List[str]] format.")

    @staticmethod
    def check_type(input_name, input_obj, type):
        if not isinstance(input_obj, type):
            raise TypeError("`" + input_name + "` is required of type " + str(type))
