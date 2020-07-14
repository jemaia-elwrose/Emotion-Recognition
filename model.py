import math
import copy
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from attentions import MaskedEdgeAttention
from modules import CNNFeatureExtractor, DialogueRNN, GraphNetwork


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    
    """

    all_perms = set()
    array = np.arange(l)
    
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
            
        all_perms = all_perms.union(perms)
        
    return list(all_perms)


def batch_graphify(features, qmask, lengths, window_past, window_future,
                   edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network.
    Pytorch geometric puts all nodes for classification in one single graph.
    Following this, we create a single graph for a mini-batch of dialogue instances.
    This method ensures that the various graph indexing is properly carried out so as
    to make sure that, utterances (nodes) from each dialogue instance will have edges
    with utterances in that same dialogue instance, but not with utternaces from any
    other dialogue instances in that mini-batch.
    
    """
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []
    
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    
    # scores are the edge weights
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
    
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
        
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
        
            if item1[0] < item1[1]:
                # edge_type.append(0) # ablation by removing speaker dependency: only 2 relation types
                # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                # edge_type.append(1) # ablation by removing speaker dependency: only 2 relation types
                # ablation by removing temporal dependency: M^2 relation types
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    #if torch.cuda.is_available():
    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths 


class DialogueGCN_DailyModel(nn.Module):
    """
    Specific to daily dialogue dataset and it uses CNN to extract features
    from 300-dimension vector to 100-dimension vector.
    
    """
    
    def __init__(self, base_model='GRU',
                D_m=100, D_g=150, D_p=150, D_e=100, D_h=100, D_a=100, graph_hidden_size=100,
                n_speakers=2, max_seq_len=70,
                window_past=7, window_future=7,
                vocab_size=18180, embedding_dim=100,
                cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3, 4, 5), cnn_dropout=0.2,
                n_classes=7, listener_state=False, context_attention='general',
                dropout_rec=0.3, dropout=0.3,
                nodal_attention=False, avec=False, no_cuda=False):

        super(DialogueGCN_DailyModel, self).__init__()
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters,
                                                      cnn_kernel_sizes, cnn_dropout)
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        # The base model is the sequential context encoder.
        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                            context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                            context_attention, D_a, dropout_rec)
            
        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)

        else:
            print('Base model must be one of DialogueRNN/LSTM/GRU')
            raise NotImplementedError

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention

        self.graph_net = GraphNetwork(2 * D_e, n_classes, n_relations, max_seq_len, graph_hidden_size, dropout,
                                      self.no_cuda)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping


    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)


    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        
        """
        
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, input_seq, qmask, umask, seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        
        """
        
        U = self.cnn_feat_extractor(input_seq, umask)

        if self.base_model == "DialogRNN":

            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)

            else:
                # seq_len, batch, D_e
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)

        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)

        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions, qmask, seq_lengths,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask, self.nodal_attention,
                                  self.avec)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths
