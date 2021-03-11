import numpy as np
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from models.mutils import UNK_IDX, soft_embed


def build_decoder(G_class, GRU_args, deconv_args, **common_args):
    if G_class == 'gru':
        cur_args = GRU_args.copy()
        cur_args.update(common_args)
        decoder = GRUDecoder(**cur_args)
    elif G_class == 'deconv':
        cur_args = deconv_args.copy()
        cur_args.update(common_args)
        decoder = DeconvDecoder(**cur_args)
    else:
        raise ValueError('Please use one of the following for dec_type: gru | deconv.')
    return decoder


class GRUDecoder(nn.Module):
    """
    Decoder is GRU with FC layers connected to last hidden unit
    """

    def __init__(self,
                 embedding,
                 emb_dim,
                 output_dim,
                 # input_zc_dim,
                 h_dim,
                 p_word_dropout,
                 p_out_dropout,
                 skip_connetions):
        super(GRUDecoder, self).__init__()
        # Reference to word embedding
        self.emb = embedding
        self.rnn = nn.GRU(emb_dim, h_dim,
                          batch_first=True)
        # self.init_hidden_fc = nn.Linear(input_zc_dim, h_dim) # TODO use this for initial hidden state?
        self.fc = nn.Sequential(
            nn.Dropout(p_out_dropout),
            nn.Linear(h_dim, output_dim))
        self.word_dropout = WordDropout(p_word_dropout)

        self.skip_connetions = skip_connetions
        if self.skip_connetions:
            self.skip_weight_x = nn.Linear(h_dim, h_dim, bias=False)
            self.skip_weight_z = nn.Linear(h_dim, h_dim, bias=False)

    def init_hidden(self, z, c):
        return torch.cat([z, c], dim=1)

    def forward(self, x, z, c):
        mbsize, seq_len = x.shape
        # mbsize x (z_dim + c_dim)  -- required shape (num_layers * num_directions, batch, hidden_size)
        init_h = self.init_hidden(z, c)
        # zc = torch.cat([z, c], dim=1)
        # init_h = self.init_hidden_fc(zc)
        # TODO to make this work, forward_sample() has to be done carefully;
        # the init_h transform only has to be done once, not at every timestep!

        # Apply word dropout and embed
        # mbsize x seq_len x emb_dim
        dec_inputs = self.emb(self.word_dropout(x))

        # mbsize x seq_len x (z_dim + c_dim)
        expanded_init_h = init_h.unsqueeze(1).expand(-1, seq_len, -1)

        # Construct input to RNN
        # TODO parametrize whether to concat z,c to decoder input or not?
        dec_inputs = torch.cat([dec_inputs, expanded_init_h], 2)

        # Compute outputs. # mbsize x seq_len x h_dim
        rnn_out, _ = self.rnn(dec_inputs, init_h.unsqueeze(0))

        # apply skip connection
        if self.skip_connetions:
            rnn_out = self.skip_weight_x(rnn_out) + self.skip_weight_z(expanded_init_h)

        y = self.fc(rnn_out)
        return y

    def forward_sample(self, sampleSoft, sampleHard, z, c, h):
        if sampleSoft is not None:
            # with sampleSoftIx (mbsize x vocabsize) gradients will pass through
            emb = soft_embed(self.emb, sampleSoft)
        else:
            # with sampleIx (mbsize) indextensor, gradients dont pass through.
            emb = self.emb(sampleHard)
        # mb x (embdim + zdim + cdim)
        emb = torch.cat([emb, z, c], 1)
        # insert seqlen 1 (mbsize x 1 x ezcdim)
        emb = emb.unsqueeze(1)
        # 1 x mbsize x h_dim
        output, h = self.rnn(emb, h)
        #     mbsize x h_dim
        output = output.squeeze(1)

        # apply skip connection
        if self.skip_connetions:
            latent_code = torch.cat([z, c], 1)
            output = self.skip_weight_x(output) + self.skip_weight_z(latent_code)

        # [mbsize x self.n_vocab]   
        logits = self.fc(output)
        return logits, h


class WordDropout(nn.Module):
    def __init__(self, p_word_dropout):
        super(WordDropout, self).__init__()
        self.p = p_word_dropout

    def forward(self, x):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p, size=tuple(data.size()))
                .astype('uint8')
        ).to(x.device)

        mask = mask.bool()
        # Set to <unk>
        data[mask] = UNK_IDX

        return data


class DeconvDecoder(nn.Module):
    """
    Implements a Deconvolutional Decoder.
    """

    def __init__(self,
                 embedding,
                 output_dim,
                 h_dim,
                 max_seq_len,
                 num_filters=300,
                 kernel_size=4,
                 num_deconv_layers=3,
                 useRNN=False,
                 temperature=1.0,
                 use_batch_norm=True,
                 num_conv_layers=2,
                 add_final_conv_layer=True,
                 emb_dim=None):
        # , p_word_dropout = 0, p_out_dropout = 0):
        """
        Params:
            `h_dim`: dimensionality of the prior z + class label c.
            `max_seq_len`: sentence length.
            `output_dim`: vocabulary size.
            `num_filters`: number of filters used in the deconvolutions.
            `num_hu_fc`: number of units in the fully connected layer.
            `kernel_size`: kernel size.
            `num_deconv_layers`: number of deconv layers.
            `useRNN`: if True, an RNN will be applied to the output of the CNN.
            `temperature`: temperatur term to be used before applying softmax.
            `use_batch_norm`: if True, batch norm is added after each (de)conv. layer.
            `num_conv_layers`: number of convolutional layers used before the last deconv layer.
            `add_final_conv_layer`: if True, a convolutional layer with kernel size 7 
                                    is used after the last deconv layer.
        
        OBS: for max_seq_len < 13, use kernel_size < 5 and num_deconv_layers == 3
        """
        super(DeconvDecoder, self).__init__()

        self.useRNN = useRNN
        self.temperature = temperature
        self.add_final_conv_layer = add_final_conv_layer
        self.last_gen_logits = []
        self.last_gen_logits_pointer = 0
        embedding_size = embedding.weight.size(1)

        if num_deconv_layers > 4:
            num_deconv_layers = 4
            print("Maximum number of deconv layers is 4.")

        if max_seq_len < 30 and kernel_size > 3:
            # for segments smaller than 30, we can only have 3 deconvolutional layers when kernel size is > 3
            num_deconv_layers = 3

        # computes the sentence size for each layer of the generator
        sentence_size_per_layer = [max_seq_len - 1]
        for i in range(num_deconv_layers - 1):
            sentence_size_per_layer.append(
                int(math.floor((sentence_size_per_layer[-1] - kernel_size) / 2) + 1)
            )
        sentence_size_per_layer.reverse()

        nnLayers = OrderedDict()
        # From: (mb, nz, 1, 1),  To: (mb, num_filters * 2, sentence_size_per_layer[0], 1)
        nnLayers["deconv_%d" % (len(nnLayers))] = nn.ConvTranspose2d(h_dim, num_filters * 2,
                                                                     (sentence_size_per_layer[0], 1), stride=2)
        if use_batch_norm:
            nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(num_filters * 2)
        nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()

        # From: (mb, num_filters * 2, sentence_size_per_layer[0], 1),  To: (mb, num_filters, sentence_size_per_layer[
        # 1], 1)
        nnLayers["deconv_%d" % (len(nnLayers))] = nn.ConvTranspose2d(num_filters * 2, num_filters, (kernel_size, 1),
                                                                     stride=2, output_padding=(1, 0))
        if use_batch_norm:
            nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(num_filters)
        nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()

        for i in range(num_conv_layers):
            # From: (mb, num_filters, sentence_size_per_layer[1], 1),  To: (mb, num_filters, sentence_size_per_layer[
            # 1], 1)
            nnLayers["conv_%d" % (len(nnLayers))] = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1,
                                                              padding=(1, 0), bias=False)
            if use_batch_norm:
                nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(num_filters)
            nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()

        if num_deconv_layers > 3:
            # From: (mb, num_filters, sentence_size_per_layer[1], 1),  To: (mb, num_filters, sentence_size_per_layer[
            # 2], 1)
            nnLayers["deconv_%d" % (len(nnLayers))] = nn.ConvTranspose2d(num_filters, num_filters, (kernel_size, 1),
                                                                         stride=2, output_padding=(1, 0))
            if use_batch_norm:
                nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(num_filters)
            nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()

        # From: (mb, num_filters, sentence_size_per_layer[-2], 1),  To: (mb, 1, sentence_size_per_layer[-1],
        # embedding_size)
        nnLayers["deconv_%d" % (len(nnLayers))] = nn.ConvTranspose2d(num_filters, 1, (kernel_size, embedding_size),
                                                                     stride=2, output_padding=(1, 0))
        nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(1)

        if add_final_conv_layer:
            nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()
            # From: (mb, (mb, 1, sentence_size_per_layer[-1], embedding_size),  To: (mb, embedding_size,
            # sentence_size_per_layer[-1], 1)
            nnLayers["conv_%d" % (len(nnLayers))] = nn.Conv2d(1, embedding_size, (7, embedding_size), stride=1,
                                                              padding=(3, 0))
            nnLayers["btn_%d" % (len(nnLayers))] = nn.BatchNorm2d(embedding_size)

        if self.useRNN:
            nnLayers["relu_%d" % (len(nnLayers))] = nn.ReLU()
            self.rnn = nn.GRU(embedding_size, embedding_size)

        self.cnn = nn.Sequential(nnLayers)

        self.fc = nn.Sequential(
            nn.Linear(embedding_size, output_dim)
        )

    def parameters(self):
        """
        Outputs the set of parameters of this nn.Model.
        """
        parametersG = set()
        parametersG |= set(self.cnn.parameters())
        if self.useRNN:
            parametersG |= set(self.rnn.parameters())
        parametersG |= set(self.fc.parameters())
        return parametersG

    def init_hidden(self, z, c):
        self.last_gen_logits = self.forward(None, z, c)
        self.last_gen_logits_pointer = 0
        return torch.cat([z, c], dim=1)

    def forward(self, x, z, c):
        """
        Params:
            `z`: latent code. Dimensionality: [mbsize x z_dim] 
            `c`: class label. Dimensionality: [mbsize x c_dim]

        Outputs soft one-hot representations. 
            Dimensionality: (minibatch_size, max_seq_len, vocabulary_size)
        """
        # [mbsize x (z_dim + c_dim)]
        latent_code = torch.cat([z, c], dim=1)
        # FROM [mbsize x (z_dim + c_dim)] to  [mbsize x (z_dim + c_dim) x 1 x 1]
        latent_code = latent_code.unsqueeze(2).unsqueeze(3)

        # dec_sent.size(): (minibatch_size, 1, max_seq_len, embedding_size)
        dec_sent = self.cnn(latent_code)

        if self.add_final_conv_layer:
            # dec_sent.size() is as follows when the final conv layer is ued: 
            # (minibatch_size, embedding_size, max_seq_len, 1)
            dec_sent = dec_sent.permute(0, 3, 2, 1).contiguous()

        minibatch_size, _, max_seq_len, embedding_size = dec_sent.size()

        if self.useRNN:
            # input to rnn must be (max_seq_len, minibatch_size, embedding_size)
            inputToRNN = dec_sent.squeeze(1).permute(1, 0, 2)
            dec_sent, _ = self.rnn(inputToRNN)
            # goes back to (minibatch_size, max_seq_len, embedding_size)
            dec_sent = dec_sent.permute(1, 0, 2).contiguous()

        dec_sent = dec_sent.view(minibatch_size * max_seq_len, embedding_size)

        # [mbsize * seq_len x vocab_size]
        out_logits = self.fc(dec_sent) / self.temperature

        # [mbsize x seq_len x vocab_size]
        out_logits = out_logits.view(minibatch_size, max_seq_len, -1)
        #         # [seq_len x mbsize x vocab_size]
        #         out_logits = out_logits.permute(1, 0, 2).contiguous()

        return out_logits

    def forward_sample(self, sampleSoft, sampleHard, z, c, h):
        assert len(self.last_gen_logits) > 0
        assert self.last_gen_logits_pointer < self.last_gen_logits.size(1)

        # [mbsize x vocab_size]
        next_token = self.last_gen_logits[:, self.last_gen_logits_pointer, :]
        self.last_gen_logits_pointer += 1
        return next_token, h
