import unittest
import torch

from sacreeos.scst import Scst
from demo.data.sample_coco_refss import refss


class InputTest(unittest.TestCase):

    def test_init_exceptions(self):
        """ Test exceptions raised by wrong init input. """

        my_eos_tok = '<eos>'
        my_bos_tok = '<sos>'
        refss_with_eos = []
        for refs in refss:
            refss_with_eos.append([ref + ' ' + my_eos_tok for ref in refs])

        verbose=False

        # not including eos in standard mode
        self.assertRaises(ValueError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_D,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=my_eos_tok,
                          corpus_refss=refss, verbose=verbose)

        # including eos in No<Eos> mode
        self.assertRaises(ValueError, Scst, scst_class=Scst.SCST_CONFIG_NO_EOS,
                          metric_class=Scst.METRIC_CIDER_D,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=my_eos_tok,
                          corpus_refss=refss_with_eos, verbose=verbose)

        # again, with different args
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_NO_EOS,
                          metric_class=Scst.METRIC_BLEU,
                          base_class=Scst.BASE_AVERAGE,
                          eos_token=my_eos_tok,
                          corpus_refss=refss_with_eos, verbose=verbose)

        # omitting No Eos argument
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY, verbose=verbose)

        # provide wrong eos_token
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=None,
                          verbose=verbose)
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=5,
                          verbose=verbose)
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=['eos'],
                          verbose=verbose)

        # provide bad corpus refs
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=my_eos_tok,
                          corpus_refss=[[0]], verbose=verbose)
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=my_eos_tok,
                          corpus_refss=['hello'], verbose=verbose)
        # format is correct, but it Eos token missing
        self.assertRaises(ValueError, Scst, scst_class=Scst.SCST_CONFIG_STANDARD,
                          metric_class=Scst.METRIC_CIDER_R,
                          base_class=Scst.BASE_GREEDY,
                          eos_token=my_eos_tok,
                          corpus_refss=[['hello']], verbose=verbose)

        # in case of BLEU 'corpus_refss' Must be None
        self.assertRaises(TypeError, Scst, scst_class=Scst.SCST_CONFIG_NO_EOS,
                          metric_class=Scst.METRIC_BLEU,
                          base_class=Scst.BASE_AVERAGE,
                          eos_token=my_eos_tok,
                          corpus_refss=[['Cool']], verbose=verbose)

    def test_standard_scst_loss_exceptions(self):
        """ Test exceptions raised by wrong loss computation input. """

        my_eos_tok = 'EOS'
        my_bos_tok = 'SOS'
        refss_with_eos = []
        for refs in refss:
            refss_with_eos.append([ref + ' ' + my_eos_tok for ref in refs])


        fake_nspi = 3
        base_args = {'nspi': fake_nspi}

        scst_1 = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                      metric_class=Scst.METRIC_CIDER_D,
                      base_class=Scst.BASE_GREEDY,
                      eos_token=my_eos_tok,
                      corpus_refss=refss_with_eos,
                      base_args=base_args)

        fake_bs = 2
        fake_max_seq_len = 10
        torch_preds = -torch.abs(torch.randn(fake_bs, fake_nspi, fake_max_seq_len))  # simulate logprobs

        # correct case
        my_preds_1 = [['a dog running in the park EOS', 'a dog is running in a park EOS',
                       'a little dog chasing a ball EOS'],
                      ['a train on the tracks EOS', 'a train is passing through the road EOS',
                       'a train is passing by EOS']]
        my_refss_1 = [['a dog chasing in the park EOS', 'a dog is jumping on a park EOS',
                       'a little dog is eating ice cream EOS'],
                      ['a train is going very fast EOS', 'a train passing through cars EOS',
                       'a yellow and blue train EOS']]
        my_greedy_1 = [['a dog chasing in a lush green field EOS', 'a dog is flying through the skies EOS',
                        'a little ambitious brown doggy EOS'],
                       ['a train is departing from the station EOS', 'a train EOS',
                        'a long yellow train on the railways EOS']]
        scst_1.compute_scst_loss(sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                                 refss=my_refss_1, base_preds=my_greedy_1,
                                 eos_pos_upthresh=fake_max_seq_len,
                                 reduction='mean')

        # remove EOS in one of the sampled sentences
        my_preds_2 = [['a dog running in the park EOS', 'a dog is running in a park EOS',
                       'a little dog chasing a ball'],
                      ['a train on the tracks EOS', 'a train is passing through the road EOS',
                       'a train is passing by EOS']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_2, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')
        # remove EOS from one of the reference sentences
        my_refss_2 = [['a dog chasing in the park ', 'a dog is jumping on a park EOS',
                       'a little dog is eating ice cream EOS'],
                      ['a train is going very fast EOS', 'a train passing through cars EOS',
                       'a yellow and blue train EOS']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                          refss=my_refss_2, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')
        # remove EOS from one of the greedy sentences
        my_greedy_2 = [['a dog chasing in a lush green field EOS', 'a dog is flying through the skies EOS',
                        'a little ambitious brown doggy EOS'],
                       ['a train is departing from the station EOS', 'a train EOS',
                        'a long yellow train on the railways']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_2,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')

        # Provide the wrong dimension
        my_preds_a = ['a dog running in the park EOS', 'a dog is running in a park EOS',
                      'a little dog chasing a ball EOS',
                      'a train on the tracks EOS', 'a train is passing through the road EOS',
                      'a train is passing by EOS']
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_a, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')

        # this should work fine, reference can have varying dimensions
        my_refss_a = [['a dog is jumping on a park EOS',
                       'a little dog is eating ice cream EOS'],
                      ['a train is going very fast EOS', 'a train passing through cars EOS',
                       'a yellow and blue train EOS']]
        scst_1.compute_scst_loss(sampled_preds=my_preds_1,
                                 sampled_logprobs=torch_preds,
                                 refss=my_refss_a, base_preds=my_greedy_1,
                                 eos_pos_upthresh=fake_max_seq_len,
                                 reduction='mean')

        my_greedy_a = [['a dog is flying through the skies EOS',
                        'a little ambitious brown doggy EOS'],
                       ['a train is departing from the station EOS', 'a train EOS',
                        'a long yellow train on the railways EOS']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss,sampled_preds=my_preds_1,
                          sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_a,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')

        scst_2 = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                      metric_class=Scst.METRIC_CIDER_D,
                      base_class=Scst.BASE_AVERAGE,
                      eos_token=my_eos_tok,
                      corpus_refss=refss_with_eos,
                      base_args=base_args)

        # expect None in base_preds
        self.assertRaises(TypeError, scst_2.compute_scst_loss, sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')


    def test_noeos_scst_loss_exceptions(self):
        """ Test exceptions raised by wrong loss computation input. """

        my_eos_tok = 'EOS'
        my_bos_tok = 'BOS'

        fake_nspi = 3
        base_args = {'nspi': fake_nspi}

        scst_1 = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                      metric_class=Scst.METRIC_CIDER_D,
                      base_class=Scst.BASE_GREEDY,
                      eos_token=my_eos_tok,
                      corpus_refss=refss,
                      base_args=base_args)

        fake_bs = 2
        fake_max_seq_len = 10
        torch_preds = -torch.abs(torch.randn(fake_bs, fake_nspi, fake_max_seq_len))  # simulate logprobs

        # correct case
        my_preds_1 = [['a dog running in the park', 'a dog is running in a park',
                       'a little dog chasing a ball'],
                      ['a train on the tracks', 'a train is passing through the road',
                       'a train is passing by']]
        my_refss_1 = [['a dog chasing in the park', 'a dog is jumping on a park',
                       'a little dog is eating ice cream'],
                      ['a train is going very fast', 'a train passing through cars',
                       'a yellow and blue train']]
        my_greedy_1 = [['a dog chasing in a lush green field', 'a dog is flying through the skies',
                        'a little ambitious brown doggy'],
                       ['a train is departing from the station', 'a train',
                        'a long yellow train on the railways']]
        scst_1.compute_scst_loss(sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                                 refss=my_refss_1, base_preds=my_greedy_1,
                                 eos_pos_upthresh=fake_max_seq_len,
                                 reduction='mean')

        # add EOS in one of the sampled sentences
        my_preds_2 = [['a dog running in the park', 'a dog is running in a park',
                       'a little dog chasing a ball'],
                      ['a train on the tracks EOS', 'a train is passing through the road',
                       'a train is passing by']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_2, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')
        # add EOS in one of the reference sentences
        my_refss_2 = [['a dog chasing in the park ', 'a dog is jumping on a park',
                       'a little dog is eating ice cream'],
                      ['a train is going very fast', 'a train passing through cars EOS',
                       'a yellow and blue train']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                          refss=my_refss_2, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')
        # add EOS one of the greedy sentences
        my_greedy_2 = [['a dog chasing in a lush green field', 'a dog is flying through the skies',
                        'a little ambitious brown doggy EOS'],
                       ['a train is departing from the station', 'a train',
                        'a long yellow train on the railways']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_1, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_2,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')

        # Provide the wrong dimension
        my_preds_a = ['a dog running in the park', 'a dog is running in a park',
                      'a little dog chasing a ball',
                      'a train on the tracks', 'a train is passing through the road',
                      'a train is passing by']
        self.assertRaises(ValueError, scst_1.compute_scst_loss, sampled_preds=my_preds_a, sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_1,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')

        # this should work fine, reference can have varying dimensions
        my_refss_a = [['a dog is jumping on a park',
                       'a little dog is eating ice cream'],
                      ['a train is going very fast', 'a train passing through cars',
                       'a yellow and blue train']]
        scst_1.compute_scst_loss(sampled_preds=my_preds_1,
                                 sampled_logprobs=torch_preds,
                                 refss=my_refss_a, base_preds=my_greedy_1,
                                 eos_pos_upthresh=fake_max_seq_len,
                                 reduction='mean')

        my_greedy_a = [['a dog is flying through the skies',
                        'a little ambitious brown doggy'],
                       ['a train is departing from the station', 'a train',
                        'a long yellow train on the railways']]
        self.assertRaises(ValueError, scst_1.compute_scst_loss,sampled_preds=my_preds_1,
                          sampled_logprobs=torch_preds,
                          refss=my_refss_1, base_preds=my_greedy_a,
                          eos_pos_upthresh=fake_max_seq_len,
                          reduction='mean')
