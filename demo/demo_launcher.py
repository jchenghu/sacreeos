""" Small demonstration of Scst. """

import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR

from sacreeos.scst import Scst
from demo.net.model import ToyModel
from demo.data.dataset import ToyDataset, ToyCollator

from demo.data.sample_coco_refss import refss


class Demo(object):
    """ Small demonstration of sacreeos package usage   """
    def __init__(self):
        super(Demo, self).__init__()

        # load pre-trained model and elements
        with open('data/demo_data.pickle', 'rb') as f:
            self.save = pickle.load(f)
        self.vocab_w2i = self.save['vocab_word2idx']
        self.vocab_i2w = self.save['vocab_idx2word']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Testing on device:" + str(self.device))

        self.my_eos_tok = self.vocab_i2w[self.save['eos_idx']]
        self.my_bos_tok = self.vocab_i2w[self.save['sos_idx']]
        self.ref_text = self.save['train_text']
        self.features = self.save['train_feat']

        self.train_set = ToyDataset(self.features, self.ref_text)
        self.collate_fn = ToyCollator(self.vocab_i2w, self.vocab_w2i, self.save['unk_idx'],
                                      self.save['sos_idx'], self.save['eos_idx'])
        self.train_loader = DataLoader(self.train_set, collate_fn=self.collate_fn, batch_size=128, shuffle=True)

        self.verbose = False

        with open('../demo/data/coco_train_refss.pickle', 'rb') as f:
            self.coco_refss = pickle.load(f)

        self.coco_refss_with_eos = [[self.my_bos_tok + ' ' + ref + ' ' + self.my_eos_tok for ref in refs]
                                    for refs in self.coco_refss]
        self.refss_with_eos = [[self.my_bos_tok + ' ' + ref + ' ' + self.my_eos_tok for ref in refs]
                               for refs in refss]

    def check_model_results(self, model):
        num_artifacts = 0
        with torch.no_grad():
            model.eval()
            for _, (feats, enc_pads, refss) in enumerate(self.train_loader):
                feats = feats.float().to(self.device)
                preds_text, _ = model.get_sampled_preds(enc_x=feats, enc_pads=enc_pads, num_outputs=1,
                                                        sos_idx=self.save['sos_idx'],
                                                        eos_idx=self.save['eos_idx'],
                                                        max_len=20,
                                                        include_eos=True,
                                                        mode='max')
                preds_i2w = [' '.join([self.vocab_i2w[word] for word in seq[0]]) for seq in preds_text]

                for pred in preds_i2w:
                    splitted_pred = pred.split(' ')
                    if splitted_pred[-1] == self.my_eos_tok and (splitted_pred[-2] == 'a'
                                                                 or splitted_pred[-2] == 'of'
                                                                 or splitted_pred[-2] == 'and'):
                        num_artifacts += 1
        if num_artifacts == 0:
            print("No artifacts.")
        else:
            print("There are " + str(num_artifacts) + " artifacts.")

    def train(self, sacreEos, model, include_eos):
        optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1)
        sched = LambdaLR(optim, lr_lambda=lambda it: 5e-4)

        # while the num epochs are set reasonably high, it is still possible
        # at early epochs for the standard scst to generate no<eos> type of artifacts.
        # But for the purpose of testing we leave it to...
        num_epochs = 30
        tt = tqdm(range(num_epochs))
        for epoch in tt:
            tt.set_description("Training epoch %d" % epoch)
            for _, (feats, enc_pads, refss) in enumerate(self.train_loader):

                if not include_eos:
                    # remove eos from refss in case of No<Eos> mode
                    refss = [[r.replace(' ' + self.my_eos_tok, '') for r in refs] for refs in refss]

                feats = feats.float().to(self.device)
                nspi = 5  # num samples per input
                sample_max_len = 20
                preds_text, preds_probs = model.get_sampled_preds(enc_x=feats, enc_pads=enc_pads,
                                                                  num_outputs=nspi,
                                                                  sos_idx=self.save['sos_idx'],
                                                                  eos_idx=self.save['eos_idx'],
                                                                  max_len=sample_max_len,
                                                                  mode='sample',
                                                                  include_eos=include_eos)

                preds_i2w = [[' '.join([self.vocab_i2w[word] for word in seq]) for seq in text_blocks]
                             for text_blocks in preds_text]

                if sacreEos.get_base_class() == Scst.BASE_AVERAGE:
                    loss = sacreEos.compute_scst_loss(sampled_preds=preds_i2w, sampled_logprobs=preds_probs,
                                                      refss=refss, base_preds=None,
                                                      eos_pos_upthresh=sample_max_len,
                                                      reduction='mean')
                elif sacreEos.get_base_class() == Scst.BASE_GREEDY:
                    model.eval()
                    greedy_text, _ = model.get_sampled_preds(enc_x=feats, enc_pads=enc_pads, num_outputs=1,
                                                             sos_idx=self.save['sos_idx'],
                                                             eos_idx=self.save['eos_idx'],
                                                             max_len=sample_max_len,
                                                             mode='max',
                                                             include_eos=include_eos)

                    greedy_i2w = [[' '.join([self.vocab_i2w[word] for word in seq]) for seq in text_blocks] * nspi
                                  for text_blocks in greedy_text]

                    model.train()
                    loss = sacreEos.compute_scst_loss(sampled_preds=preds_i2w, sampled_logprobs=preds_probs,
                                                      refss=refss, base_preds=greedy_i2w,
                                                      eos_pos_upthresh=sample_max_len,
                                                      reduction='mean')
                else:
                    raise ValueError("Invald base class")

                loss.backward()
                optim.step()
                optim.zero_grad()
                sched.step()

    def test_standard_scst(self):
        scst = Scst(scst_class=Scst.SCST_CONFIG_STANDARD,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=self.my_eos_tok,
                    corpus_refss=self.coco_refss_with_eos,
                    verbose=self.verbose)
        print("Standard Scst signature: " + str(scst.get_signature()))
        toy_model = ToyModel(vocab_size=len(self.vocab_i2w), d_model=128,
                             max_len=self.save['max_len'], device=self.device)
        toy_model.load_state_dict(self.save['model_state'])
        toy_model.to(self.device)

        self.train(scst, toy_model, include_eos=True)
        self.check_model_results(toy_model)

    def test_noeos_scst(self):
        scst = Scst(scst_class=Scst.SCST_CONFIG_NO_EOS,
                    metric_class=Scst.METRIC_CIDER_D,
                    base_class=Scst.BASE_AVERAGE,
                    eos_token=self.my_eos_tok,
                    corpus_refss=self.coco_refss,
                    verbose=self.verbose)
        print("NoEos scst signature: " + str(scst.get_signature()))
        toy_model = ToyModel(vocab_size=len(self.vocab_i2w), d_model=128,
                             max_len=self.save['max_len'], device=self.device)
        toy_model.load_state_dict(self.save['model_state'])
        toy_model.to(self.device)
        self.train(scst, toy_model, include_eos=False)
        self.check_model_results(toy_model)


if __name__ == "__main__":
    print("Demo launched")
    demo = Demo()
    print("Standard Scst training started...")
    demo.test_standard_scst()
    print("Note: because of the poor train set, the standard case may present few artifacts as well")
    print("NoEos Scst training started...")
    demo.test_noeos_scst()
    print("End.")
