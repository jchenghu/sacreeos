from typing import List

from sacreeos.scst_conf import ScstConfig
from sacreeos.scst_conf.conf import ScstConf
from sacreeos.err.commonerr import CommonErrors
from sacreeos.err.commonechecks import CommonChecks


class NoEosScst(ScstConf):
    """ No<Eos> SCST arguments controller.

    The No<Eos> mode requires the special `eos_token` to be omitted in both corpus document frequencies
    calculation and training step. This will likely lead to a higher score in n-gram based metrics, but on the
    downside results will very likely suffer from trivial word fragments ending of the sentences,
    such as `with a`, 'of`, `on a`.

    :param eos_token: str : end of sequence token
    :param is_initialized: bool : set the SCST signature according to whether it used initialization or not
    """
    def __init__(self, eos_token, is_initialized):
        super().__init__(signature='NO<EOS>',
                         is_initialized=is_initialized,
                         scst_config=ScstConfig.NOEOS_CONFIG)
        self.eos_token = eos_token

    def input_check(self, seq_name, seqss: List[List[str]]):
        """ Check the input correctness accoding to the No<Eos> configuration.

        Ensures the `eos_token` is not included in the sentences

        :param seq_name: str : sequences name
        :param seqss: list of list of sentences to be checked
        :return:
        """
        CommonChecks.check_list_list_sentences_format(seq_name, seqss)
        for seqs in seqss:
            for seq in seqs:
                split_seq = seq.split(' ')
                if self.eos_token == split_seq[-1]:
                    raise ValueError("Eos Token must not be the last token in `" + seq_name + "`: " + str(seq) + ".")

    def cider_init_check(self, corpus_refss):
        """ Check the initialization corpus is correctly formed according to the No<Eos> configuration.

        Ensures the `eos_token` is not included in the initialization corpus.

        :param corpus_refss: list of list of ground-truth sentences that defines the initialization corpus
        :return:
        """
        flattened_refss = ' '.join([' '.join(refs) for refs in corpus_refss]) + ' '
        if ' ' + self.eos_token + ' ' in flattened_refss:
            raise ValueError("\n\tThe `eos_token` was found in the `corpus_refss`, this is prohibited in No<Eos> "
                             "mode. \n\tYou might be looking for the Standard scst mode instead.")

    def get_eos_token(self):
        """ Get end of sequence token.
        :return: str
        """
        return self.eos_token
