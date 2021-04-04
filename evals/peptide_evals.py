import math
import random
from math import log
import collections

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist


class PeptideEvaluator():
    def __init__(self, orig_filename=None, seq_len=0):
        self.orig_filename = orig_filename
        self.seq_len = seq_len
        self.gap_open = -10
        self.gap_extend = -1
        self.matrix = matlist.blosum62
        self.scales = {'Eisenberg': {'A': 0.25, 'R': -1.80, 'N': -0.64,
                                     'D': -0.72, 'C': 0.04, 'Q': -0.69,
                                     'E': -0.62, 'G': 0.16, 'H': -0.40,
                                     'I': 0.73, 'L': 0.53, 'K': -1.10,
                                     'M': 0.26, 'F': 0.61, 'P': -0.07,
                                     'S': -0.26, 'T': -0.18, 'W': 0.37,
                                     'Y': 0.02, 'V': 0.54},
                       }
        self._supported_scales = list(self.scales.keys())

        self.aa_charge = {'E': -1, 'D': -1, 'K': 1, 'R': 1}

        print('initialized eval_peptide class')

    def f(self, s):
        s = s.split()

        if bool(set(s) & set(["<unk>"])):
            # Remove if <unk> appears
            return ""
        if bool(set(s) & set(["<pad>"])):
            # Remove if <pad> appears
            return ""
        if bool(set(s) & set(["<start>"])):
            # Remove if <start> appears
            return ""
        if bool(set(s) & set(["<eos>"])):
            # Remove if <eos> appears
            return ""

        s = "".join(s)
        s = s.replace(" ", "")
        return s

    def convert_to_fasta(self, inpfile, fastafile, seq_len):
        fileInput = open(inpfile, "r")
        count = 1
        with open(fastafile, 'w+') as fileOutput:
            for strLine in fileInput:
                if strLine[0:5] != 'label':
                    # Strip the endline character from each input line
                    strLine = strLine.rstrip("\n").split(',')[0].replace(" ", "")

                    if 0 < len(strLine) < seq_len:
                        # Output the header
                        fileOutput.write(">" + ' ' + str(count) + ' ' + str(
                            len(strLine)) + "\n")
                        fileOutput.write(strLine + "\n")
                        count = count + 1

        fileInput.close()
        return fileOutput

    def log2(self, number):
        return log(number) / log(2)

    def assign_hydrophobicity(self, sequence, scale='Eisenberg'):
        """
        Assigns a hydrophobicity value to each amino acid in the sequence
        """

        hscale = self.scales.get(scale, None)
        if not hscale:
            raise KeyError('{} is not a supported scale. '.format(scale))

        hvalues = []
        for aa in sequence:
            sc_hydrophobicity = hscale.get(aa, None)
            if sc_hydrophobicity is None:
                raise KeyError('Amino acid not defined in scale: {}'.format(aa))
            hvalues.append(sc_hydrophobicity)

        return hvalues

    def calculate_moment(self, array, angle=100):
        """
        Calculates the hydrophobic dipole moment from an array of hydrophobicity
        values. Formula defined by Eisenberg, 1982 (Nature). Returns the average
        moment (normalized by sequence length)

        uH = sqrt(sum(Hi cos(i*d))**2 + sum(Hi sin(i*d))**2),
        where i is the amino acid index and d (delta) is an angular value in
        degrees (100 for alpha-helix, 180 for beta-sheet).
        """

        sum_cos, sum_sin = 0.0, 0.0
        for i, hv in enumerate(array):
            rad_inc = ((i * angle) * math.pi) / 180.0
            sum_cos += hv * math.cos(rad_inc)
            sum_sin += hv * math.sin(rad_inc)
        return math.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(array)

    def calculate_charge(self, sequence):
        """Calculates the charge of the peptide sequence at pH 7.4
        """
        sc_charges = [self.aa_charge.get(aa, 0) for aa in sequence]
        return sum(sc_charges)

    def heuristics(self, seqs):
        polar_aa = set(('S', 'T', 'N', 'H', 'Q', 'G'))
        speci_aa = set(('P', 'C'))
        apolar_aa = set(('A', 'L', 'V', 'I', 'M'))
        charged_aa = set(('E', 'D', 'K', 'R'))
        aromatic_aa = set(('W', 'Y', 'F'))
        all_aas = collections.defaultdict(int)

        resultsdict = {}
        aa_count = 0
        nlines = 0
        z, av_h, av_uH = 0, 0, 0
        n_p, n_s, n_a, n_ar, n_c = 0, 0, 0, 0, 0

        for rec in seqs:
            rec = self.f(rec)
            nlines = nlines + 1
            aa_count = aa_count + len(str(rec))  # count of all aas in corpus
            x = ProteinAnalysis(str(rec))
            for aa, count in x.count_amino_acids().items():
                all_aas[aa] += count
                if aa in polar_aa:
                    n_p += count
                elif aa in speci_aa:
                    n_s += count
                elif aa in apolar_aa:
                    n_a += count
                elif aa in charged_aa:
                    n_c += count
                elif aa in aromatic_aa:
                    n_ar += count

            z += self.calculate_charge(str(rec))
            seq_h = self.assign_hydrophobicity(str(rec))
            av_h += sum(seq_h) / len(seq_h)
            av_uH += self.calculate_moment(seq_h)

        av_h = av_h / len(seqs)  # avg_hydrophobicity over entire corpus
        av_uH = av_uH / len(seqs)  # avg hydrophobic moment over entire corpus

        sizes = [len(rec) for rec in seqs]
        avg_size = float((sum(sizes)) / len(sizes))

        av_n_p = round(n_p / aa_count, 3)
        av_n_s = round(n_s / aa_count, 3)
        av_n_a = round(n_a / aa_count, 3)
        av_n_c = round(n_c / aa_count, 3)
        av_n_ar = round(n_ar / aa_count, 3)

        resultsdict = {'av_h': av_h, 'av_uH': av_uH, 'avg_size': avg_size,
                       'av_n_p': av_n_p,
                       'av_n_s': av_n_s, 'av_n_a': av_n_a, 'av_n_c': av_n_c,
                       'av_n_ar': av_n_ar}

        return resultsdict

    def aa_composition(self, seqs):
        all_aas = collections.defaultdict(int)

        aa_count = 0
        nlines = 0

        for rec in seqs:
            rec = self.f(rec)
            nlines = nlines + 1
            aa_count = aa_count + len(str(rec))
            x = ProteinAnalysis(str(rec))
            for aa, count in x.count_amino_acids().items():
                all_aas[aa] += count

        if aa_count < 1:
            return {'A': 1, 'R': 1, 'N': 1, 'D': 1, 'C': 1, 'Q': 1,
                    'E': 1, 'G': 1, 'H': 1, 'I': 1, 'L': 1, 'K': 1,
                    'M': 1, 'F': 1, 'P': 1, 'S': 1, 'T': 1, 'W': 1,
                    'Y': 1, 'V': 1}

        aa_countsdict = {}
        for aa in all_aas:
            count = round(all_aas[aa] / aa_count, 3)
            aa_countsdict[aa] = count

        return aa_countsdict

    def similarity(self, seqs_lst1, seqs_lst2, matrix_size=100):
        resultsdict = {}
        sim_lst = []

        for rec in random.sample(seqs_lst1, matrix_size):
            for rec1 in random.sample(seqs_lst2, matrix_size):
                rec = self.f(rec)
                rec1 = self.f(rec1)
                if len(rec) > 1:
                    if len(rec1) > 1:
                        if str(rec) != str(rec1):
                            alns = pairwise2.align.globalds(str(rec),
                                                            str(rec1),
                                                            self.matrix,
                                                            self.gap_open,
                                                            self.gap_extend)
                            top_aln = alns[0]
                            al1, al2, score, begin, end = top_aln
                            sim_lst.append(score / (log(len(rec))))

        resultsdict['sim'] = sim_lst
        av_sim = sum(sim_lst) / len(sim_lst) if len(sim_lst) > 0 else 0.0

        return resultsdict, av_sim
