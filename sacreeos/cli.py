
from sacreeos.metric import MetricClass
from sacreeos.metric import METRIC_QUESTION_TEXT, METRIC_POSSIBLE_ANS, METRIC_DEFAULT_ANS, METRIC_HELP_TEXT
from sacreeos.metric.cider import CiderBase
from sacreeos.metric.cider_d import CiderD
from sacreeos.metric.cider_r import CiderR

from sacreeos.scst_conf import ScstConfig
from sacreeos.scst_conf import SCSTINIT_QUESTION_TEXT, SCSTINIT_POSSIBLE_ANS, SCSTINIT_DEFAULT_ANS, \
                               SCSTINIT_HELP_TEXT, \
                               SCSTMODE_QUESTION_TEXT, SCSTMODE_POSSIBLE_ANS, SCSTMODE_HELP_TEXT

from sacreeos.reward_base import BaseRewardClass
from sacreeos.reward_base import BASE_QUESTION_TEXT, BASE_POSSIBLE_ANS, BASE_DEFAULT_ANS, BASE_HELP_TEXT, \
                                 BASE_ARGS_QUESTION_TEXT, BASE_ARGS_DEFAULT_ANS, \
                                 BASE_ARGS_HELP_TEXT

from sacreeos.utils.cli_args import Str2PositiveType

from sacreeos.scst import Scst


class ManualSignature(object):
    """ Manual Signature Generator.    """
    
    # States
    METRIC_SELECTION_STATE = 0
    METRIC_ARGS_SELECTION_STATE = 1
    SCSTINIT_SELECTION_STATE = 2
    SCSTMODE_SELECTION_STATE = 3
    BASE_SELECTION_STATE = 4
    BASE_ARGS_SELECTION_STATE = 5
    GOT_ALL_ANSWERS_STATE = 6
    
    MAX_NUM_STATES = 7

    HELP_KEY = 'h'
    PREVIOUS_KEY = 'p'
    QUIT_KEY = 'x'
    OTHER_POSSIBLE_ANS_FIRST = {HELP_KEY: 'help', QUIT_KEY: 'quit'}
    OTHER_POSSIBLE_ANS = {HELP_KEY: 'help', PREVIOUS_KEY: 'previous', QUIT_KEY: 'quit'}

    # aestethics
    INCOMPLETE_MARK = '✕'
    COMPLETED_MARK = '✓'
    NOT_RECOGNIZED_TEXT = "Answer not recognized, please select one of the available options."
    QUIT_TEXT = "Signature generation interrupted. Closing the program."
    ANSWER_SELECTED_TEXT = "Selected answer:"
    DEFAULT_SELECTED_TEXT = "Default selected:"

    def __init__(self, interactive=True, inputs_queue=None, verbose=True):
        if not interactive and inputs_queue is None:
            raise ValueError("'inputs_queue' must be a list of inputs and cannot be None if interactive "
                             "is set to False")

        self.verbose = verbose

        self.interactive = interactive
        self.inputs_queue = inputs_queue
        if not interactive:
            inputs_queue.reverse()  # reverse input so it becomes a fifo queue. pop() returns the first input

        self.state = self.METRIC_SELECTION_STATE
        self.state_counter = 1

        self.answers = {self.SCSTINIT_SELECTION_STATE: None,
                        self.SCSTMODE_SELECTION_STATE: None,
                        self.METRIC_SELECTION_STATE: None,
                        self.METRIC_ARGS_SELECTION_STATE: None,
                        self.BASE_SELECTION_STATE: None,
                        self.BASE_ARGS_SELECTION_STATE: None}

        self.selected_metric = None

    @staticmethod
    def get_progress_bar(num_completed, num_incomplete):
        return '[' + ''.join([ManualSignature.COMPLETED_MARK]) * num_completed + \
               ''.join([ManualSignature.INCOMPLETE_MARK]) * num_incomplete + ']'

    @staticmethod
    def print_beautiful_dict(dict):
        beautified_print = '('
        num_items = len(dict.items())
        for item, i in zip(dict.items(), range(num_items)):
            key, value = item
            beautified_print += str(key) + '=' + "\'" + str(value) + "\'"
            last_item = i == num_items-1
            if not last_item:
                beautified_print += '/ '
        beautified_print += ')'
        return beautified_print

    def print_if_verbose(self, string, end=None):
        if self.verbose:
            if end is not None:
                print(string, end=end)
            else:
                print(string)

    def welcome(self):
        self.print_if_verbose("\nWelcome to the manual SacreEOS signature generation interface.")
        self.print_if_verbose("Press the letters indicated in the round brakets and ENTER.")
        self.print_if_verbose("Press ENTER only to select the default answer.\n")

    def get_answer(self, question, accepted_answers):
        if self.interactive:
            user_input = input(question + ' ')
        else:
            user_input = self.inputs_queue.pop()
        if accepted_answers is None:
            # user can type anything, no input check
            return user_input

        # check if it is an accepted answer
        if (user_input is '') or (user_input in accepted_answers):
            return user_input
        else:
            lower_ans = str.lower(user_input)
            for key, value in accepted_answers.items():
                lower_key = str.lower(key)
                lower_value = str.lower(value)
                if lower_ans == lower_value or lower_ans == lower_key:
                    return key
        # if not, notify the caller with None
        return None

    def get_question(self):
        if self.state == self.METRIC_SELECTION_STATE:
            return METRIC_QUESTION_TEXT
        elif self.state == self.METRIC_ARGS_SELECTION_STATE:
            if self.selected_metric is None:
                raise RuntimeError("Metric is not selected yet!")
            elif self.selected_metric == MetricClass.CIDEr_D:
                return CiderD.QUESTION_TEXT
            elif self.selected_metric == MetricClass.CIDEr_R:
                return CiderR.QUESTION_TEXT
            elif self.selected_metric == MetricClass.CIDEr_Base:
                return CiderBase.QUESTION_TEXT
            elif self.selected_metric == MetricClass.BLEU:
                raise NotImplementedError("For the moment it doesn't expect arguments to " +
                                          "be different than the default for the BLEU score.")
            else:
                raise RuntimeError("Unexpected metric selection.")
        elif self.state == self.SCSTINIT_SELECTION_STATE:
            return SCSTINIT_QUESTION_TEXT
        elif self.state == self.SCSTMODE_SELECTION_STATE:
            return SCSTMODE_QUESTION_TEXT
        elif self.state == self.BASE_SELECTION_STATE:
            return BASE_QUESTION_TEXT
        elif self.state == self.BASE_ARGS_SELECTION_STATE:
            return BASE_ARGS_QUESTION_TEXT
        else:
            raise RuntimeError("Unexpected state.")

    def update_state(self):
        # max_num_states '- 1' <- removes the closing program state from the progress counting
        question = ManualSignature.get_progress_bar(self.state_counter,
                                                    (self.MAX_NUM_STATES-1) - self.state_counter) + ' '
        question += self.get_question() + ' '

        if self.state == self.METRIC_SELECTION_STATE:
            question += self.print_beautiful_dict({**METRIC_POSSIBLE_ANS,
                                                   **ManualSignature.OTHER_POSSIBLE_ANS_FIRST})
            ans = self.get_answer(question, {**METRIC_POSSIBLE_ANS,
                                             **ManualSignature.OTHER_POSSIBLE_ANS_FIRST})
            if ans is None:
                if self.verbose:
                    self.print_if_verbose(self.NOT_RECOGNIZED_TEXT)
                return
            elif ans in METRIC_POSSIBLE_ANS.keys():
                self.print_if_verbose(ManualSignature.ANSWER_SELECTED_TEXT + ' ' + METRIC_POSSIBLE_ANS[ans], end='\n\n')
                self.selected_metric = MetricClass.answer_to_class(ans)
                self.answers[self.METRIC_SELECTION_STATE] = self.selected_metric
            elif ans == '':
                self.print_if_verbose(ManualSignature.DEFAULT_SELECTED_TEXT + ' ' +
                                      str(METRIC_POSSIBLE_ANS[METRIC_DEFAULT_ANS]), end='\n\n')
                self.selected_metric = MetricClass.answer_to_class(METRIC_DEFAULT_ANS)
                self.answers[self.METRIC_SELECTION_STATE] = self.selected_metric
            elif ans == ManualSignature.HELP_KEY:
                self.print_if_verbose(METRIC_HELP_TEXT)
                return  # preserve the state
            elif ans == ManualSignature.QUIT_KEY:
                self.print_if_verbose(ManualSignature.QUIT_TEXT)
                exit(0)

            if self.selected_metric == MetricClass.BLEU or self.selected_metric == MetricClass.CIDEr_Base:
                self.state = ManualSignature.SCSTINIT_SELECTION_STATE
                self.state_counter += 2  # currently it does not support different aguments for these two scores
            else:
                self.state = self.METRIC_ARGS_SELECTION_STATE
                self.state_counter += 1
            return
        elif self.state == self.METRIC_ARGS_SELECTION_STATE:
            # nested loop for each metric
            if self.selected_metric == MetricClass.CIDEr_D:
                arg_name_list = CiderD.get_args_name_list_for_manual()
                questions_list = CiderD.get_questions_text_for_manual()
                default_list = CiderD.get_args_default_list_for_manual()
                input_convert_list = CiderD.get_args_data_convert_list_for_manual()
            elif self.selected_metric == MetricClass.CIDEr_R:
                arg_name_list = CiderR.get_args_name_list_for_manual()
                questions_list = CiderR.get_questions_text_for_manual()
                default_list = CiderR.get_args_default_list_for_manual()
                input_convert_list = CiderR.get_args_data_convert_list_for_manual()
            elif self.selected_metric == MetricClass.CIDEr_Base:
                arg_name_list = CiderR.get_args_name_list_for_manual()
                questions_list = CiderBase.get_questions_text_for_manual()
                default_list = CiderBase.get_args_default_list_for_manual()
                input_convert_list = CiderBase.get_args_data_convert_list_for_manual()
            elif self.selected_metric == MetricClass.BLEU:
                raise RuntimeError("BLEU does not expect custom options (in this current implementation).")

            self.print_if_verbose(question + '\n')
            args_dict = {}
            for name, qus, default, str2data in zip(arg_name_list, questions_list,
                                                    default_list, input_convert_list):
                accepted_answer = False
                while not accepted_answer:
                    if self.interactive:
                        ans = input(qus + ' ')
                    else:
                        ans = self.inputs_queue.pop()
                    if ans == '':
                        args_dict[name] = default
                        self.print_if_verbose("Default selected: " + str(default))
                    elif str2data(ans) != Str2PositiveType.CONVERSION_FAILED:
                        args_dict[name] = str2data(ans)
                        self.print_if_verbose("Selected: " + str(ans))
                    else:
                        self.print_if_verbose("Not accepted input")
                        continue
                    accepted_answer = True

            self.print_if_verbose('', end='\n')

            try:
                # try construction
                if self.selected_metric == MetricClass.CIDEr_D:
                    CiderD(**args_dict)
                elif self.selected_metric == MetricClass.CIDEr_R:
                    CiderR(**args_dict)
                elif self.selected_metric == MetricClass.CIDEr_Base:
                    CiderBase(**args_dict)
                else:
                    raise RuntimeError("Unexpected class")
            except Exception as e:
                self.print_if_verbose("\nArguments not accepted due to the following error: " + str(e))
                self.print_if_verbose("Please re-insert arguments.", end='\n\n')
                return

            self.answers[ManualSignature.METRIC_ARGS_SELECTION_STATE] = args_dict
            self.state = self.SCSTINIT_SELECTION_STATE
            self.state_counter += 1
            return

        elif self.state == self.SCSTINIT_SELECTION_STATE:
            question += self.print_beautiful_dict({**SCSTINIT_POSSIBLE_ANS,
                                                              **ManualSignature.OTHER_POSSIBLE_ANS})
            ans = self.get_answer(question, {**SCSTINIT_POSSIBLE_ANS,
                                  **ManualSignature.OTHER_POSSIBLE_ANS})
            if ans is None:
                self.print_if_verbose(self.NOT_RECOGNIZED_TEXT)
                return
            elif ans in SCSTINIT_POSSIBLE_ANS.keys():
                self.answers[self.SCSTINIT_SELECTION_STATE] = SCSTINIT_POSSIBLE_ANS[ans]
                self.print_if_verbose(ManualSignature.ANSWER_SELECTED_TEXT + ' ' + SCSTINIT_POSSIBLE_ANS[ans], end='\n\n')
            elif ans == '':
                self.print_if_verbose(ManualSignature.DEFAULT_SELECTED_TEXT + ' ' +
                                      str(SCSTINIT_POSSIBLE_ANS[SCSTINIT_DEFAULT_ANS]), end='\n\n')
                self.answers[self.SCSTINIT_SELECTION_STATE] = SCSTINIT_POSSIBLE_ANS[SCSTINIT_DEFAULT_ANS]
            elif ans == ManualSignature.HELP_KEY:
                self.print_if_verbose(SCSTINIT_HELP_TEXT, end='\n\n')
                return
            elif ans == ManualSignature.PREVIOUS_KEY:
                self.state = self.METRIC_SELECTION_STATE
                self.state_counter -= 2
                self.print_if_verbose('', end="\n\n")
                return  # go back previous state
            elif ans == ManualSignature.QUIT_KEY:
                self.print_if_verbose(ManualSignature.QUIT_TEXT)
                exit(0)

            self.state = self.SCSTMODE_SELECTION_STATE
            self.state_counter += 1
            return

        elif self.state == self.SCSTMODE_SELECTION_STATE:
            question += self.print_beautiful_dict({**SCSTMODE_POSSIBLE_ANS,
                                                              **ManualSignature.OTHER_POSSIBLE_ANS})
            ans = self.get_answer(question, {**SCSTMODE_POSSIBLE_ANS,
                                  **ManualSignature.OTHER_POSSIBLE_ANS})
            if ans is None:
                self.print_if_verbose(self.NOT_RECOGNIZED_TEXT)
                return
            elif ans in SCSTMODE_POSSIBLE_ANS.keys():
                self.answers[self.SCSTMODE_SELECTION_STATE] = SCSTMODE_POSSIBLE_ANS[ans]
                self.print_if_verbose(ManualSignature.ANSWER_SELECTED_TEXT + ' ' + SCSTMODE_POSSIBLE_ANS[ans], end='\n\n')
            elif ans == '' or ans == ManualSignature.HELP_KEY:
                self.print_if_verbose(SCSTMODE_HELP_TEXT, end='\n\n')
                return
            elif ans == ManualSignature.PREVIOUS_KEY:
                self.state = self.SCSTINIT_SELECTION_STATE
                self.state_counter -= 1
                self.print_if_verbose('', end="\n\n")
                return
            elif ans == ManualSignature.QUIT_KEY:
                self.print_if_verbose(ManualSignature.QUIT_TEXT, end='\n\n')
                exit(0)

            self.state = self.BASE_SELECTION_STATE
            self.state_counter += 1
            return

        elif self.state == self.BASE_SELECTION_STATE:
            question += self.print_beautiful_dict({**BASE_POSSIBLE_ANS,
                                                   **ManualSignature.OTHER_POSSIBLE_ANS})
            ans = self.get_answer(question, {**BASE_POSSIBLE_ANS,
                                  **ManualSignature.OTHER_POSSIBLE_ANS})
            if ans is None:
                self.print_if_verbose(self.NOT_RECOGNIZED_TEXT)
                return
            elif ans in BASE_POSSIBLE_ANS.keys():
                self.answers[self.BASE_SELECTION_STATE] = BASE_POSSIBLE_ANS[ans]
                self.print_if_verbose(ManualSignature.ANSWER_SELECTED_TEXT + ' ' + BASE_POSSIBLE_ANS[ans], end='\n\n')
            elif ans == '':
                self.print_if_verbose(ManualSignature.DEFAULT_SELECTED_TEXT + ' ' +
                                      str(BASE_POSSIBLE_ANS[BASE_DEFAULT_ANS]), end='\n\n')
                self.answers[self.BASE_SELECTION_STATE] = BASE_POSSIBLE_ANS[BASE_DEFAULT_ANS]
            elif ans == ManualSignature.HELP_KEY:
                self.print_if_verbose(BASE_HELP_TEXT, end='\n\n')
                return
            elif ans == ManualSignature.PREVIOUS_KEY:
                self.state = self.SCSTMODE_SELECTION_STATE
                self.state_counter -= 1
                self.print_if_verbose('', end="\n\n")
                return
            elif ans == ManualSignature.QUIT_KEY:
                self.print_if_verbose(ManualSignature.QUIT_TEXT, end='\n\n')
                exit(0)

            self.state = self.BASE_ARGS_SELECTION_STATE
            self.state_counter += 1
            return

        elif self.state == self.BASE_ARGS_SELECTION_STATE:
            # Little hack: we expect a numeric answer and is indicated in #PTR_KQ"8
            # whereas the possible answers data contain only "movement" keys
            question += self.print_beautiful_dict(ManualSignature.OTHER_POSSIBLE_ANS)
            ans = self.get_answer(question, None)

            if ans is None:
                self.print_if_verbose(self.NOT_RECOGNIZED_TEXT, end='\n\n')
                return
            if ans.isnumeric():
                ans = int(ans)
                if ans >= 1:
                    self.answers[self.BASE_ARGS_SELECTION_STATE] = ans
                    self.print_if_verbose(ManualSignature.ANSWER_SELECTED_TEXT + ' ' + str(ans), end='\n\n')
            else:
                if ans == '':
                    self.print_if_verbose(ManualSignature.DEFAULT_SELECTED_TEXT + ' ' + str(BASE_ARGS_DEFAULT_ANS), end='\n\n')
                    self.answers[self.BASE_ARGS_SELECTION_STATE] = BASE_ARGS_DEFAULT_ANS
                elif ans.isalpha():
                    if ans == ManualSignature.HELP_KEY:
                        self.print_if_verbose(BASE_ARGS_HELP_TEXT, end='\n\n')
                        return
                    elif ans == ManualSignature.PREVIOUS_KEY:
                        self.state = self.BASE_SELECTION_STATE
                        self.state_counter -= 1
                        self.print_if_verbose('', end="\n\n")
                        return
                    elif ans == ManualSignature.QUIT_KEY:
                        self.print_if_verbose(ManualSignature.QUIT_TEXT, end='\n\n')
                        exit(0)
                    else:
                        self.print_if_verbose("Invalid option.")
                        return
                else:
                    self.print_if_verbose("The input is neither a character nor an integer.")
                    return

            self.state = self.GOT_ALL_ANSWERS_STATE
            return

        else:
            raise RuntimeError("Unexpected state.")

    def generate_signature(self):
        self.welcome()

        # Get data from user
        while self.state != self.GOT_ALL_ANSWERS_STATE:
            self.update_state()

        # convert answers to the interface classes of Scst
        if self.answers[ManualSignature.METRIC_SELECTION_STATE] == MetricClass.CIDEr_D:
            scst_metric_class = Scst.METRIC_CIDER_D
        elif self.answers[ManualSignature.METRIC_SELECTION_STATE] == MetricClass.CIDEr_R:
            scst_metric_class = Scst.METRIC_CIDER_R
        elif self.answers[ManualSignature.METRIC_SELECTION_STATE] == MetricClass.CIDEr_Base:
            scst_metric_class = Scst.METRIC_CIDER
        elif self.answers[ManualSignature.METRIC_SELECTION_STATE] == MetricClass.BLEU:
            scst_metric_class = Scst.METRIC_BLEU
        else:
            raise RuntimeError("Selected metric name not recognized.")
        
        if self.answers[ManualSignature.SCSTMODE_SELECTION_STATE] == ScstConfig.SELECTION_YES:
            scst_scstmode_class = ScstConfig.STANDARD_CONFIG
        elif self.answers[ManualSignature.SCSTMODE_SELECTION_STATE] == ScstConfig.SELECTION_NO:
            scst_scstmode_class = ScstConfig.NOEOS_CONFIG
        else:
            raise RuntimeError("Selected scst mode not expected.")
        
        if self.answers[ManualSignature.BASE_SELECTION_STATE] == BaseRewardClass.GREEDY_BASE_STRING_ID:
            scst_base_class = Scst.BASE_GREEDY
        elif self.answers[ManualSignature.BASE_SELECTION_STATE] == BaseRewardClass.AVERAGE_BASE_STRING_ID:
            scst_base_class = Scst.BASE_AVERAGE
        else:
            raise RuntimeError("Selected base reward not recognized.")

        # dummy data for the sake of signature generation
        if self.answers[ManualSignature.SCSTINIT_SELECTION_STATE] == ScstConfig.SELECTION_YES:
            if self.answers[ManualSignature.SCSTMODE_SELECTION_STATE] == ScstConfig.SELECTION_YES:
                dummy_train_ref = [['<bos> a dog catching a frisbee in the park <eos>'],
                                   ['<bos> a cat sleeping next to a teddy bear <eos>']]
            else:
                dummy_train_ref = [['<bos> a dog catching a frisbee in the park'],
                                   ['<bos> a cat sleeping next to a teddy bear']]
        elif self.answers[ManualSignature.SCSTINIT_SELECTION_STATE] == ScstConfig.SELECTION_NO:
            dummy_train_ref = None

        # self.print_if_verbose(self.answers)
        scst = Scst(scst_class=scst_scstmode_class,
                    metric_class=scst_metric_class,
                    base_class=scst_base_class,
                    metric_args=self.answers[ManualSignature.METRIC_ARGS_SELECTION_STATE],
                    eos_token='<eos>',
                    corpus_refss=dummy_train_ref,
                    base_args={'nspi': self.answers[ManualSignature.BASE_ARGS_SELECTION_STATE]},
                    verbose=False)
        self.print_if_verbose("Here is your SacreEOS signature:\n" + str(scst.get_signature()) + '\n')
        return scst.get_signature()


