import torch
import torch.nn as nn
from utils import check_dir_exists

UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
EOS_IDX = 3


def save_model(model, fn):
    check_dir_exists(fn)
    torch.save(model.state_dict(), fn)
    print('Saved model to ' + fn)


# check_mask_eos: no tokens past EOS in a generated sentence
# input sentence: 1, length of sentence
# returns padded sentence: 1, length of sentence
# return index of first padded token
def check_mask_eos(sentence, model):
    sentence.squeeze_(0)  # remove leading singleton dim
    eos_ix = (sentence == model.EOS_IDX).nonzero().squeeze()
    assert eos_ix.nelement() in [0, 1], 'expecting NO or SINGLE occurence of eos'
    eos_ix = eos_ix.item() if eos_ix.nelement() == 1 else sentence.size(0)
    all_pad_beyond = (sentence[eos_ix + 1:] == model.PAD_IDX).all()
    assert all_pad_beyond, 'BUG. there shouldnt be junk behind eos. See issue #10 and commit 1d1b1d2 '
    return eos_ix


def onehot_embed(hardIx, vocabSize):
    """ Get tensor hardIx (mbsize), return it's one hot embedding  (mbsize x vocabSize) """
    assert hardIx.dim() == 1, 'expecting 1D tensor: minibatch of indices.'
    softIx = torch.zeros(hardIx.size(0), vocabSize).to(hardIx.device)
    softIx.scatter_(1, hardIx.unsqueeze(1), 1.0)
    return softIx


def soft_embed(embed, softIx):
    assert isinstance(embed, nn.Embedding), 'Expecting nn.Embedding'
    # NOTE embedding weight has to be transposed, stored differently for embedding
    # out1 = F.linear(softIx, embed.weight.t(), bias=None)
    out2 = softIx @ embed.weight  # MMult: [mbsize x vocab] * [vocab x emb_dim]
    # assert torch.allclose(out1, out2)
    return out2
