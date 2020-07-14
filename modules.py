import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv

from attentions import MatchingAttention, SimpleAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):

        var = tensor
        if length > var.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var

    else:
        if length > tensor.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the
    graph convoluted features, as in Equation 4, 5, 6. in the paper.
    
    """
    
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    
    #if torch.cuda.is_available():
    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda) 
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:,0,:])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer,
                           dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    
    """

    if nodal_attn:
        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        
        return log_prob

    else:
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        
        return log_prob


class DialogueRNNCell(nn.Module):
    """
    DialogueRNNCell for DialogueRNN module.
    
    """

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        
        self.g_cell = nn.GRUCell(D_m+D_p, D_g)
        self.p_cell = nn.GRUCell(D_m+D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)

        self.listener_state = listener_state
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)


    def _select_parties(self, X, indices):
        
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
            
        q0_sel = torch.cat(q0_sel, dim=0)
        
        return q0_sel
    

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        
        """
        
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                         torch.zeros(U.size()[0],self.D_g).type(U.type()) \
                         if g_hist.size()[0]==0 else g_hist[-1])
        g_ = self.dropout(g_)
        
        if g_hist.size()[0]==0:
            
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
            
        # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
        #         else self.attention(g_hist,U)[0] # batch, D_g
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
            
        else:
            ql_ = q0
            
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0 else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        
        return g_, q_, e_, alpha


class DialogueRNN(nn.Module):
    """
    DialogueRNN module.
    
    """

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e, listener_state,
                                             context_attention, D_a, dropout)
        

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        
        """

        # 0-dimensional tensor
        g_hist = torch.zeros(0).type(U.type())
        # batch, party, D_p
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2], self.D_p).type(U.type())
        # batch, D_e
        e_ = torch.zeros(0).type(U.type())
        e = e_

        alpha = []
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        # seq_len, batch, D_e
        return e,alpha


class LSTMModel(nn.Module):
    """
    LSTM base model or sequential context encoder.
    
    """

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2,
                            bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)


    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
                
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
            
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        
        return log_prob, alpha, alpha_f, alpha_b, emotions


class GRUModel(nn.Module):
    """
    GRU base model or sequential context encoder.
    
    """

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2,
                          bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
                
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
            
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        
        return log_prob, alpha, alpha_f, alpha_b, emotions


class GraphNetwork(torch.nn.Module):
    """
    The Speaker-level context encoder in the form of a 2 layer GCN.
    
    """
    
    def __init__(self, num_features, num_classes, num_relations, max_seq_len,
                 hidden_size=64, dropout=0.5, no_cuda=False):
        super(GraphNetwork, self).__init__()
        
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features+hidden_size,
                                          num_features+hidden_size, att_type='general2')
        
        self.linear   = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc  = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda 

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        
        out = self.conv1(x, edge_index, edge_type)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        
        log_prob = classify_node_features(emotions, seq_lengths, umask,
                                          self.matchatt, self.linear, self.dropout,
                                          self.smax_fc, nodal_attn, avec, self.no_cuda)
        
        return log_prob


class CNNFeatureExtractor(nn.Module):
    """
    Module from DialogueRNN tp extract features 300-dimension vector to 100-dimension vector.
    
    """
    
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=filters,
                                              kernel_size=K) for K in kernel_sizes])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size


    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        
        # Setting requires grad to false
        self.embedding.weight.requires_grad = False
        

    def forward(self, x, umask):
        """
        x -> num_utt, batch, num_words
        umask -> batch, num_utt
        
        """

        num_utt, batch, num_words = x.size()

        # (num_utt, batch, num_words)
        x = x.type(torch.LongTensor).to(device)
        
        # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        x = x.view(-1, num_words)

        # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = self.embedding(x)
        # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)
        emb = emb.transpose(-2, -1).contiguous()

        # Perform convolution
        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        # Perform maxpooling 1d
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, dim=1)

        # Linear layer
        # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = F.relu(self.fc(self.dropout(concated)))
        # (num_utt * batch, 100) -> (num_utt, batch, 100)
        features = features.view(num_utt, batch, -1)

        # (batch, num_utt) -> (batch, num_utt, 1)
        mask = umask.unsqueeze(-1).type(torch.FloatTensor).to(device)
        # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.transpose(0, 1)
        # (num_utt, batch, 1) -> (num_utt, batch, 100)
        mask = mask.repeat(1, 1, self.feature_dim)
        # (num_utt, batch, 100) -> (num_utt, batch, 100)
        features = (features * mask)

        return features


class DialogRNNModel(nn.Module):
    """
    DialogRNNModel module.

    """
    def __init__(self, D_m, D_g, D_p, D_e, D_h, D_a=100, n_classes=7, listener_state=False,
                context_attention='simple', dropout_rec=0.5, dropout=0.5):
        super(DialogRNNModel, self).__init__()

        self.dropout   = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout+0.15)
        
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e,listener_state,
                                    context_attention, D_a, dropout_rec)

        self.matchatt = MatchingAttention(2*D_e,2*D_e,att_type='general2')
        self.linear     = nn.Linear(2*D_e, D_h)
        self.smax_fc    = nn.Linear(D_h, n_classes)


    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """

        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party

        """

        # seq_len, batch, D_e
        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
        emotions_f = self.dropout_rec(emotions_f)

        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)

        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f,emotions_b],dim=-1)

        # alpha, alpha_f, alpha_b = [], [], []
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])

            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))

        else:
            hidden = F.relu(self.linear(emotions))

        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)

        # seq_len, batch, n_classes
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        
        return log_prob, alpha, alpha_f, alpha_b, emotions
