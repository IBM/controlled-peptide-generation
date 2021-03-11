import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier(classifier_type, emb_dim, **C_args):
    # TODO: IF/ELSE over other classifier types
    if classifier_type == 'cnn':
        classifier = CNNClassifier(emb_dim, **C_args)
    else:
        raise ValueError('Please use CNN classifier')
    return classifier


class CNNClassifier(nn.Module):
    """
    Sequence classifier based on a CNN architecture (Kim, 2014)
    """

    def __init__(self,
                 emb_dim,
                 min_filter_width,
                 max_filter_width,
                 num_filters,
                 dropout):
        super(CNNClassifier, self).__init__()
        self.max_filter_width = max_filter_width

        self.conv_layers = nn.ModuleList([nn.Conv2d(1,
                                                    num_filters,
                                                    (width, emb_dim))
                                          for width in range(min_filter_width, max_filter_width + 1)])

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * (max_filter_width - min_filter_width + 1), 2)
        )

    def forward(self, x):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
        assert x.size(2) >= self.max_filter_width, 'Current classifier arch needs at least seqlen {}'.format(
            self.max_filter_width)

        # Compute filter outputs
        features = []
        for ix, filters in enumerate(self.conv_layers):
            cur_layer = F.relu(filters(x)).squeeze(3)
            cur_pooled = F.max_pool1d(cur_layer, cur_layer.size(2)).squeeze(2)
            features.append(cur_pooled)

        # Build feature vector
        x = torch.cat(features, dim=1)

        # Compute distribution over c in output layer
        p_c = self.fc(x)

        return p_c
