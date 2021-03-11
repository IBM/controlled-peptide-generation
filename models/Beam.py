import torch


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       device (torch.device): gpu or cpu device
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1,
                 device=torch.device('cpu'),
                 min_length=0):

        self.size = size
        self.device = device

        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).to(self.device).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.LongTensor(size)
                            .to(self.device)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self._bos = bos
        self.eos_top = False

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Minimum prediction length
        self.min_length = min_length

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs):
        """
        Given prob over words for every last beam `wordLk`
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        Returns: True if beam search is complete.
        """
        assert not self.done(), 'not expecting to advance once done'
        num_words = word_probs.size(1)
        # force the output to be longer than self.min_length and to avoid BOS
        cur_len = len(self.next_ys)
        for k in range(len(word_probs)):
            if cur_len < self.min_length:
                word_probs[k][self._eos] = -1e20
            # HERE IS DIFFERENT FROM GREEDY: NEVER LET BOS AS PREDICTION THROUGH
            word_probs[k][self._bos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                          self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id // num_words

        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -2, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]
