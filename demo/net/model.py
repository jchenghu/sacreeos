
import torch
import torch.nn as nn

from demo.net.layers import PositionalEncoder, Encoder, Decoder
from demo.net.utils import create_pad_mask, create_no_peak_and_pad_mask


class ToyModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, max_len=80, device='cuda'):
        super(ToyModel, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        self.input_linear = nn.Linear(2048, d_model)
        self.enc = Encoder(d_model)
        self.dec = Decoder(d_model)

        self.pe = PositionalEncoder(d_model, max_len, device)
        self.vocab_linear = torch.nn.Linear(d_model, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_embedder = nn.Embedding(vocab_size, d_model)

    def forward_enc(self, enc_x, enc_pads):
        pad_mask = create_pad_mask(mask_size=(enc_x.size(0), enc_x.size(1), enc_x.size(1)),
                                   pad_row=enc_pads, pad_column=enc_pads).to(self.device)
        return self.enc(x=enc_x, mask=pad_mask)

    def forward_dec(self, cross_x, enc_pads, dec_y, dec_pads, apply_log_softmax=False):

        no_peak_mask = create_no_peak_and_pad_mask(mask_size=(dec_y.size(0), dec_y.size(1), dec_y.size(1)),
                                                   num_pads=dec_pads).to(self.device)

        pad_mask = create_pad_mask(mask_size=(dec_y.size(0), dec_y.size(1), cross_x.size(1)),
                                   pad_row=dec_pads, pad_column=enc_pads).to(self.device)

        y = self.out_embedder(dec_y)
        y = self.pe(y)
        y = self.dec(y, no_peak_mask, cross_x, pad_mask)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y

    def forward(self, enc_x, dec_x=None, enc_pads=[0], dec_pads=[0], apply_log_softmax=False):
        x = self.forward_enc(enc_x, enc_pads)
        y = self.forward_dec(x, enc_pads, dec_x, dec_pads, apply_log_softmax)
        return y

    def get_sampled_preds(self, enc_x, enc_pads, num_outputs, sos_idx, eos_idx, max_len,
                          mode='sample', include_eos=True):
        bs, enc_seq_len, _ = enc_x.shape

        enc_input_num_pads = [enc_pads[i] for i in range(bs) for _ in range(num_outputs)]

        x = self.forward_enc(enc_x=enc_x, enc_pads=enc_input_num_pads)
        x = x.unsqueeze(1).expand(-1, num_outputs, -1, -1).reshape(bs * num_outputs, enc_seq_len, x.shape[-1])

        upperbound_vector = torch.tensor([max_len] * bs * num_outputs, dtype=torch.int).to(self.device)
        where_is_eos_vector = upperbound_vector.clone()
        eos_vector = torch.tensor([eos_idx] * bs * num_outputs, dtype=torch.long).to(self.device)
        finished_flag_vector = torch.zeros(bs * num_outputs).type(torch.int)

        predicted_caption = torch.tensor([sos_idx] * (bs * num_outputs), dtype=torch.long).to(self.device).unsqueeze(-1)
        predicted_caption_prob = torch.zeros(bs * num_outputs).to(self.device).unsqueeze(-1)

        # no paddings are used because once an EOS is encountered the following words are not
        # actually returned in the result so no need to pad them
        dec_input_num_pads = [0]*(bs*num_outputs)
        time_step = 0
        while (finished_flag_vector.sum() != bs * num_outputs) and time_step < max_len:
            dec_input = predicted_caption
            log_probs = self.forward_dec(x, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=True)

            if mode == 'sample':
                prob_dist = torch.distributions.Categorical(torch.exp(log_probs[:, time_step]))
                sampled_word_indexes = prob_dist.sample()
            elif mode == 'max':
                _, sampled_word_indexes = torch.topk(log_probs[:, time_step], k=num_outputs, sorted=True)
                sampled_word_indexes = sampled_word_indexes.squeeze(-1)

            predicted_caption = torch.cat((predicted_caption, sampled_word_indexes.unsqueeze(-1)), dim=-1)
            predicted_caption_prob = torch.cat((predicted_caption_prob,
                log_probs[:, time_step].gather(index=sampled_word_indexes.unsqueeze(-1), dim=-1)), dim=-1)
            time_step += 1

            # update the eos position tracker
            where_is_eos_vector = torch.min(where_is_eos_vector,
                                    upperbound_vector.masked_fill(sampled_word_indexes == eos_vector, time_step))
            # the max operation implement a logic 'or' to check whether the caption is finished or not
            finished_flag_vector = torch.max(finished_flag_vector,
                                             (sampled_word_indexes == eos_vector).type(torch.IntTensor))

        # remove the elements that come after the first eos from the sequence
        res_predicted_caption = []
        for i in range(bs):
            res_predicted_caption.append([])
            for j in range(num_outputs):
                index = i*num_outputs + j
                if include_eos:
                    res_predicted_caption[i].append(
                        predicted_caption[index, :where_is_eos_vector[index].item()+1].tolist())
                else:
                    res_predicted_caption[i].append(
                        predicted_caption[index, :where_is_eos_vector[index].item()].tolist())

        # create a mask that zeroes out the element that comes after the first eos
        where_is_eos_vector = where_is_eos_vector.unsqueeze(-1).expand(-1, time_step+1)
        if include_eos:
            # do nothing, it's already included in the tensor
            pass
        else:
            # trick to discard eos logprob
            where_is_eos_vector = where_is_eos_vector - 1

        arange_tensor = torch.arange(time_step+1).unsqueeze(0).expand(bs * num_outputs, -1).to(self.device)
        predicted_caption_prob.masked_fill_(arange_tensor > where_is_eos_vector, 0.0)
        res_predicted_caption_prob = predicted_caption_prob.reshape(bs, num_outputs, -1)

        # res_predicted_caption: list of list of ints [bs, num_outpus, token indexes]
        # res_predicted_caption_prob: list of list of floats [bs, num_outputs, logprobs]
        return res_predicted_caption, res_predicted_caption_prob
