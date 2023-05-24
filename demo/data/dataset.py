
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ToyCollator(object):
    def __init__(self, vocab_i2w, vocab_w2i, unk_idx, sos_idx, eos_idx):
        self.vocab_i2w = vocab_i2w
        self.vocab_w2i = vocab_w2i
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.unk_word = vocab_i2w[unk_idx]
        self.sos_word = vocab_i2w[sos_idx]
        self.eos_word = vocab_i2w[eos_idx]

    def __call__(self, batch):
        refs = [[ref + ' ' + self.eos_word for ref in elem[0]] for elem in batch]
        feat_len = [elem[1].size(1) for elem in batch]
        enc_pads = max(feat_len) - np.array(feat_len)
        feats = pad_sequence([elem[1].squeeze(0) for elem in batch], batch_first=True)
        return feats, enc_pads.tolist(), refs


class ToyDataset(Dataset):
    """ Toy dataset for image captioning. """
    def __init__(self, features, text):
        self.features = features
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.features[idx]


