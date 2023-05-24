from typing import List

from sacreeos.scst_conf import ScstConfig
from sacreeos.scst_conf.conf import ScstConf
from sacreeos.err.commonerr import CommonErrors
from sacreeos.err.commonechecks import CommonChecks


class StandardScst(ScstConf):
    """ Standard SCST arguments controller.

    The Standard SCST requires the special `eos_token` to be included in both corpus document frequencies
    calculation and training step. This will likely lead to a lower score compared to the No<Eos> configuration,
    in case of n-gram based rewards but results won't suffer from trivial words ending.

    :param eos_token: str : end of sequence token
    :param is_initialized: bool : set the SCST signature according to whether it used initialization or not
    """
    def __init__(self, eos_token, is_initialized):
        super().__init__(signature='STANDARD',
                         is_initialized=is_initialized,
                         scst_config=ScstConfig.STANDARD_CONFIG)
        self.eos_token = eos_token

    def input_check_with_thresh(self, seq_name, seqss: List[List[str]], eos_pos_upthresh):
        """ Check the input correctness accoding to the Standard configuration.

        Ensures the `eos_token` is included in both predictions and ground truth,
        in every position of the sentence up until `eos_pos_upthresh`.

        :param seq_name: str : sequences name
        :param seqss: list of list of sentences to be checked
        :param eos_pos_upthresh: the upper threshold along the sequence length up until which the input
                                 checks regarding the `eos` are effective.
        :return:
        """
        CommonChecks.check_list_list_sentences_format(seq_name, seqss)
        for seqs in seqss:
            for seq in seqs:
                split_seq = seq.split(' ')
                if self.eos_token != split_seq[-1] and len(split_seq) < eos_pos_upthresh:
                    raise ValueError("`eos_token` must be the last token for predictions which length is smaller than "
                                     "`eos_pos_upthresh`=" + str(eos_pos_upthresh) + ".\n"
                                     "`eos_token` was not the last otken in the prediction `" + str(seq) + "`.")

    def input_check(self, seq_name, seqss: List[List[str]]):
        """ Check the input correctness accoding to the Standard configuration.

        Ensures the `eos_token` is included in the sentences

        :param seq_name: str : sequences name
        :param seqss: list of list of sentences to be checked
        :return:
        """
        CommonChecks.check_list_list_sentences_format(seq_name, seqss)
        for seqs in seqss:
            for seq in seqs:
                split_seq = seq.split(' ')
                if self.eos_token != split_seq[-1]:
                    raise ValueError("Eos Token must be the last token in `" + seq_name + "`: " + str(seq) + ".")

    def cider_init_check(self, corpus_refss: List[List[str]]):
        """ Check the initialization corpus is correctly formed according to the Standard configuration.

        Ensures the `eos_token` is included in the initialization corpus.

        :param corpus_refss: list of list of ground-truth sentences that defines the initialization corpus
        :return:
        """
        for refs in corpus_refss:
            for ref in refs:
                words = ref.split(' ')
                if words[-1] != self.eos_token:
                    raise ValueError("`eos_token` was not the last token in sentence " + str(ref) + "\n"
                                     "all sentences in `corpus_refss` should be terminated by the eos token "
                                     "in the STANDARD scst.")

    def get_eos_token(self):
        """ Get end of sequence token.
        :return: str
        """
        return self.eos_token
