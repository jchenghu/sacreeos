
import torch
from typing import List

from . import __VERSION__
from sacreeos.err.commonerr import CommonErrors
from sacreeos.metric import MetricClass
from sacreeos.metric.cider import CiderBase
from sacreeos.metric.cider_d import CiderD
from sacreeos.metric.cider_r import CiderR
from sacreeos.metric.bleu import BLEU
from sacreeos.scst_conf import ScstConfig
from sacreeos.scst_conf.standard import StandardScst
from sacreeos.scst_conf.noeos import NoEosScst
from sacreeos.reward_base import BaseRewardClass
from sacreeos.reward_base.greedy import GreedyBase
from sacreeos.reward_base.average import AverageBase


class Scst(object):

    """ User classes interface.    """
    SCST_CONFIG_STANDARD = 1
    SCST_CONFIG_NO_EOS = 2

    METRIC_CIDER = 30
    METRIC_CIDER_D = 31
    METRIC_CIDER_R = 32
    METRIC_BLEU = 33

    BASE_GREEDY = 70
    BASE_AVERAGE = 71

    """ Construct Scst.
        
        SCST Class can be selected between Scst.STANDARD and Scst.NO<EOS>_MODE, the latter typically
            outperform the first but results often suffer from trivial fragment words that increase the overall
            score without providing semantic value and downgrading the linguistic correctness. The first 
            typically does not suffer from artifacts, except in some particular circumstances such as
            poor dataset or no corpus intialization.
            
        see README.md for more details.
    """
    def __init__(self, scst_class, metric_class, base_class,

                 # path to corpus_tfidf
                 eos_token,
                 corpus_refss=None,

                 verbose=True,

                 metric_args=None,
                 base_args=None):
        """ Construct the three main components, the scst_conf, the metric and rewad base """

        self.metric_is_initialized = corpus_refss is not None
        self.scst_conf = None
        self.reward_base = None
        self.metric = None
        self.scst_class = scst_class
        self.metric_class = metric_class
        self.base_class = base_class
        self.verbose = verbose

        # construct scst
        if not isinstance(eos_token, str):
            raise TypeError("`eos_token` must be str.")
        if scst_class == Scst.SCST_CONFIG_STANDARD:
            self.scst_conf = StandardScst(eos_token, self.metric_is_initialized)
            if self.verbose:
                print("[SacreEOS] Scst info: Standard SCST mode is selected.")
        elif scst_class == Scst.SCST_CONFIG_NO_EOS:
            self.scst_conf = NoEosScst(eos_token, self.metric_is_initialized)
            if self.verbose:
                print("[SacreEOS] Scst warning: No<Eos> mode is selected, in this case, the CIDEr score "
                      "will be higher compared to the standard mode but trivial fragments such as `with a`, "
                      "`of a`, `and a` are expected in the results.")
        else:
            raise ValueError(CommonErrors.invalid_scst_class())

        # construct metric
        self.use_cider_metric = (self.metric_class == Scst.METRIC_CIDER_R
                                 or self.metric_class == Scst.METRIC_CIDER_D
                                 or self.metric_class == Scst.METRIC_CIDER)
        # check metric arguments...
        if self.use_cider_metric:
            if corpus_refss is None:
                if self.verbose:
                    print("[SacreEOS] Scst warning: `corpus_refss` is None, hence CIDEr metrics won't be initialized,"
                          " tf-idfs will be computed from the references only."
                          " Defining an initialization corpus is suggested, as it is empiricallly proven to"
                          " be more effective (such in the case of MS-COCO 2014)."
                          " Additionally, without proper initialization, even the Standard Scst may suffer"
                          " from trivial words termination artifact in the results.")
            elif not (isinstance(corpus_refss, list) and isinstance(corpus_refss[0], list)
                    and isinstance(corpus_refss[0][0], str)):
                raise TypeError("`corpus_refss` must be list of list of sentences.")
            else:
                self.scst_conf.cider_init_check(corpus_refss)
                if self.verbose:
                    print("[SacreEOS] Scst info: `corpus_refss` has been set. The likelihood of trivial words "
                          "termination artifacts will be significantly reduced (in case of "
                          "MS-COCO 2014 it is close to zero). However, in case of a poor and small train"
                          "set, it may still occasionally suffer from these artifacts.")

        # ...apply default arguments in case they any was provided...
        self.metric_args = metric_args
        if metric_args is None:
            if metric_class is Scst.METRIC_CIDER:
                self.metric_args = {'n': CiderBase.DEFAULT_N}
            elif metric_class is Scst.METRIC_CIDER_D:
                self.metric_args = {'n': CiderD.DEFAULT_N, 'sigma': CiderD.DEFAULT_SIGMA}
            elif metric_class is Scst.METRIC_CIDER_R:
                self.metric_args = {'n': CiderR.DEFAULT_N, 'repeat_coeff': CiderR.DEFAULT_REPEAT_COEFF,
                                    'length_coeff': CiderR.DEFAULT_LENGTH_COEFF, 'alpha': CiderR.DEFAULT_ALPHA}
            elif metric_class is Scst.METRIC_BLEU:
                pass  # BLEU requires no args
        elif not isinstance(metric_args, dict):
            raise TypeError("`metric_args` must be dictionary.")
        else:
            # ... or cover the unspecified arguments with the default ones
            if metric_class is Scst.METRIC_CIDER:
                self.metric_args = {'n': metric_args.get('n', CiderBase.DEFAULT_N)}
            elif metric_class is Scst.METRIC_CIDER_D:
                self.metric_args = {'n': metric_args.get('n', CiderD.DEFAULT_N),
                                    'sigma': metric_args.get('sigma', CiderD.DEFAULT_SIGMA)}
            elif metric_class is Scst.METRIC_CIDER_R:
                self. metric_args = {'n': metric_args.get('n', CiderR.DEFAULT_N),
                                     'repeat_coeff': metric_args.get('repeat_coeff', CiderR.DEFAULT_REPEAT_COEFF),
                                     'length_coeff': metric_args.get('length_coeff', CiderR.DEFAULT_LENGTH_COEFF),
                                     'alpha': metric_args.get('alpha', CiderR.DEFAULT_ALPHA)}

        if metric_class == Scst.METRIC_CIDER_D:
            self.metric = CiderD(n=self.metric_args['n'], sigma=self.metric_args['sigma'], corpus_refss=corpus_refss)
        elif metric_class == Scst.METRIC_CIDER:
            self.metric = CiderBase(n=self.metric_args['n'], corpus_refss=corpus_refss)
        elif metric_class == Scst.METRIC_CIDER_R:
            self.metric = CiderR(n=self.metric_args['n'], repeat_coeff=self.metric_args['repeat_coeff'],
                                 length_coeff=self.metric_args['length_coeff'], alpha=self.metric_args['alpha'],
                                 corpus_refss=corpus_refss)
        elif metric_class == Scst.METRIC_BLEU:
            self.metric = BLEU()
            if corpus_refss is not None:
                raise TypeError("\t`corpus_refss` must be `None` since BLEU does not expect initialization.")
        else:
            raise ValueError(CommonErrors.invalid_metric_class())

        # construct base
        if base_args is None:
            if base_class is Scst.BASE_GREEDY:
                self.base_args = {'nspi': GreedyBase.DEFAULT_NSPI}
            elif base_class is Scst.BASE_AVERAGE:
                self.base_args = {'nspi': GreedyBase.DEFAULT_NSPI}
        else:
            self.base_args = base_args

        if base_class == Scst.BASE_GREEDY:
            self.reward_base = GreedyBase(nspi=self.base_args['nspi'])
        elif base_class == Scst.BASE_AVERAGE:
            self.reward_base = AverageBase(nspi=self.base_args['nspi'])
        else:
            raise ValueError(CommonErrors.invalid_reward_base_class())

        # the loss computation method require the user a possibly complicated (hence error prone) argument
        # therefore a warning is issued once
        self.already_sent_pad_warning = False

        # print the shareable signature to the user
        if verbose:
            print("[SacreEOS] signature: " + str(self.get_signature()))

    def compute_scst_loss(self, sampled_preds: List[List[str]], sampled_logprobs: torch.tensor,
                          refss: List[List[str]], base_preds=None, eos_pos_upthresh=None,
                          reduction='mean',
                          get_stat_data=False
                          ):
        """
            Compute the SCST loss function.
            See README.md for details.
        """
        # jchu: Is all the fuss about `eos_pos_upthresh` necessary?
        #         I believe it is, if the method always looked for the eos token in the whole sequence
        #         throuhout all the training step, it will most likely kill the process during the first epochs
        #         where sampled caption may reach the max_seq_len without producing Eos. The whole purpose
        #         of the library is to prevent head-aches and fuss about this token... so I don't want to
        #         waste users precious time. Besides, setting the argument to None should work fine for most
        #         applications.
        if torch.is_tensor(sampled_logprobs):
            if self.verbose and (not self.already_sent_pad_warning):
                print("SacreEOS Scst warning:: `sampled_logprobs` argument is expected to be properly padded")
                self.already_sent_pad_warning = True

            # check data
            bs, _, max_len = sampled_logprobs.shape
            if eos_pos_upthresh is None:
                # by default assume no sub-word techniques are being adopted and do not report issues if eos is
                # missing in the last sequence position. This approach is motivated by the fact that, in the
                # Standard configuration, if the <eos> token was mistakenly omitted, it should be missing also in
                # one of the sequences which lengths aren't the maximum, hence the catching web should be
                # reasonably wide. Whereas, if the <eos> token was included, but it happens to be missing in the
                # last position, it is most likely because of an unfortunate sampling case
                eos_pos_upthresh = max_len

            if base_preds is not None and self.reward_base.get_class() == BaseRewardClass.AVERAGE:
                raise TypeError("`base_preds` argument should be None in case of Average base method.")
            if (len(sampled_preds) != bs) or (len(refss) != bs) or \
                    (base_preds is not None and (len(base_preds) != bs)):
                error_msg = "Mismatching sizes at dimension 0 should be " + str(bs) + ", instead " \
                            + "`sampled_preds` size at dim 0 is " + str(len(sampled_preds)) + ' ' \
                            + "`refss` size at dim 0 is " + str(len(refss)) + '\n'
                if base_preds is not None:
                    error_msg += "`base_preds` at dim 0 is " + str(len(base_preds)) + '\n'

                raise ValueError(error_msg)
            for preds in sampled_preds:
                if len(preds) != self.reward_base.get_nspi():
                    raise ValueError("Mismatching sizes at dimension 1 should be "
                                     + str(self.reward_base.get_nspi()) + ", instead `sampled_preds` got "
                                     + str(len(preds)) + " elements in " + str(preds) + ".")
            if base_preds is not None:
                for base in base_preds:
                    if len(base) != self.reward_base.get_nspi():
                        reminder = ''
                        if self.base_args == Scst.BASE_GREEDY:
                            reminder = "\nReminder: Greedy mode was selected, " + \
                                       "the greedy decoded sequence must be repeated nspi times."
                        raise ValueError("Mismatching sizes at dimension 1 should be "
                                         + str(self.reward_base.get_nspi()) + ", instead `base_preds` got "
                                         + str(len(base)) + " elements in " + str(base) + "." + reminder)

            if self.scst_conf.get_class() == ScstConfig.STANDARD_CONFIG:
                self.scst_conf.input_check_with_thresh('sampled_preds', sampled_preds, eos_pos_upthresh)
                if base_preds is not None:
                    self.scst_conf.input_check_with_thresh('base_preds', base_preds, eos_pos_upthresh)
                self.scst_conf.input_check('refss', refss)
            if self.scst_conf.get_class() == ScstConfig.NOEOS_CONFIG:
                self.scst_conf.input_check('sampled_preds', sampled_preds)
                self.scst_conf.input_check('refss', refss)
                if base_preds is not None:
                    self.scst_conf.input_check('base_preds', base_preds)
            self.metric.input_check(sampled_preds, refss)

            device = str(sampled_logprobs.device)
            # convert text data in the desired format
            flattened_tests = [t for samples_of_one_input in sampled_preds for t in samples_of_one_input]
            repeated_refss = [refs for refs in refss for _ in range(self.reward_base.get_nspi())]

            # compute rewards
            _, pred_reward_array = self.metric.compute(tests=flattened_tests, refss=repeated_refss)
            if self.metric.get_metric_class() == MetricClass.BLEU:
                # 4 BLEUs are returned, take only the last one
                pred_reward_array = [p[-1] for p in pred_reward_array]

            pred_reward = torch.tensor(pred_reward_array).to(device).reshape(bs, self.reward_base.get_nspi())
            if self.reward_base.get_class() == BaseRewardClass.GREEDY:
                # should we leave it to the user to repeat base predictions in case of greedy base?
                # For the time being yes, it should be less confusing
                # since the interface is the same of the average approach.

                flattened_base = [b for samples_of_one_input in base_preds for b in samples_of_one_input]

                _, base_reward_array = self.metric.compute(tests=flattened_base, refss=repeated_refss)
                if self.metric.get_metric_class() == MetricClass.BLEU:
                    # 4 bleus are provided, take only the last one
                    base_reward_array = [b[-1] for b in base_reward_array]

                base_reward = torch.tensor(base_reward_array).to(device).reshape(bs, self.reward_base.get_nspi())
                self.reward_base.input_check(pred_reward, base_reward)
                based_reward = self.reward_base.compute_based_reward(pred_reward, base_reward)
            elif self.reward_base.get_class() == BaseRewardClass.AVERAGE:
                self.reward_base.input_check(pred_reward)
                based_reward, base_reward_array = self.reward_base.compute_based_reward(pred_reward)
            else:
                raise TypeError("Base class not expected.")

            loss = based_reward * (-sampled_logprobs.sum(dim=-1))

            if reduction == 'sum':
                loss = loss.sum()
            elif reduction == 'mean':
                loss = loss.mean()
            else:  # None
                pass

            if get_stat_data:
                return loss, base_reward_array, pred_reward_array
            else:
                return loss

        else:
            raise NotImplementedError("SacreEOS error: the library currently supports only Pytorch tensors.")

    # getters
    def get_scst_class(self):
        """ Get the scst configuration class.
        :return: int
        """
        return self.scst_class

    def get_base_class(self):
        """ Get the scst reward base class.
        :return: int
        """
        return self.base_class

    def get_metric_class(self):
        """ Get the scst metric class.
        :return: int
        """
        return self.metric_class

    def get_metric_args(self):
        """ Return the metric arguments.
        :return: dictionary
        """
        return self.metric_args

    def get_base_args(self):
        """ Return the base arguments.
        :return: dictionary
        """
        return self.base_args

    def get_eos_token(self):
        """ Get end of sequence token.
        :return: str
        """
        return self.scst_conf.get_eos_token()

    def get_signature(self):
        """ Return the SacreEOS signature
        :return: str
        """
        return '+'.join([self.scst_conf.get_signature()] + [self.metric.get_signature()] +
                        [self.reward_base.get_signature()] + [__VERSION__])

