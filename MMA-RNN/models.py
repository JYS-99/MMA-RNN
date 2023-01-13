import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class HierAttention(nn.Module):
    def __init__(self, input_size):
        super(HierAttention, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(self.input_size, self.input_size)
        self.query = nn.Parameter(torch.randn(self.input_size))

    def forward(self, input):

        alpha = torch.tanh(self.fc(input))
        alpha = torch.matmul(alpha, self.query)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(dim=2)
        context_u = torch.sum(alpha * input, dim=1)
        return context_u, alpha.squeeze(dim=2)

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.input_size)

    def forward(self, input):

        alpha = torch.tanh(self.fc1(input))
        query = self.fc2(input)
        alpha = torch.matmul(query, alpha.permute(0,2,1))
        alpha = F.softmax(alpha, dim=2)
        output = torch.matmul(alpha,input)
        return output

class MMECG(nn.Module):

    def __init__(self, num_char_classes = 2, num_seq_classes = 3):
        super(MMECG, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (1, 8, 250)
        self.mean = None
        self.std = None
        self.char_hidden_size = 128
        self.word_hidden_size = 128
        self.num_layers_char = 1
        self.num_layers_word = 1
        self.bidirectional = True
        self.hidden_factor = 2 if self.bidirectional else 1
        self.char_embedding_dim = 128
        self.max_seq_len = 1500
        self.layer_norm = True

        # Modules
        self.char_dimh = self.char_hidden_size * self.hidden_factor
        self.word_dimh = self.word_hidden_size * self.hidden_factor
        self.char_rnn = nn.LSTM(self.char_embedding_dim, self.char_hidden_size, num_layers=self.num_layers_char, bidirectional=self.bidirectional,
                           batch_first=True)
        self.word_rnn = nn.LSTM(self.char_dimh, self.word_hidden_size, num_layers = self.num_layers_word,bidirectional = self.bidirectional, batch_first = True)

        self.char_embedding = nn.Linear(1, self.char_embedding_dim,bias=False)
        self.char_attention = HierAttention(self.char_dimh)
        self.word_attention = HierAttention(self.word_dimh)

        self.linear = nn.Sequential(nn.Linear(self.char_dimh, 128), nn.ReLU())
        self.dropout_layer = nn.Dropout(0.5)
        self.head2 = nn.Linear(128, num_char_classes)
        self.head1 = nn.Linear(self.word_dimh, num_seq_classes)
        torch.nn.init.xavier_uniform_(self.head1.weight)
        self.head1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.head2.weight)
        self.head2.bias.data.zero_()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers_char*self.hidden_factor, bsz, self.char_hidden_size),
                    weight.new_zeros(self.num_layers_char*self.hidden_factor,bsz, self.char_hidden_size))


    def logits(self, input, seq_len, seq_feat=None, ser_feat=None):


        #time point level

        rnn_input = rnn_utils.pad_sequence(input, batch_first=True,padding_value=0)
        rnn_input = rnn_input.unsqueeze(dim=2)
        cnt = 0
        word_inputs = []
        char_outputs = []
        char_alpha = []
        hidden = self.init_hidden(rnn_input.size(0))

        while cnt < self.max_seq_len:
            rnn_input_i = rnn_input[:, cnt:cnt + 150, :]

            rnn_input_i = self.char_embedding(rnn_input_i)
            rnn_outputs_i, hidden = self.char_rnn(rnn_input_i, hidden)

            char_outputs.append(rnn_outputs_i)
            att_outputs_i, char_alpha_i = self.char_attention(rnn_outputs_i)
            word_inputs.append(att_outputs_i)
            char_alpha.append(char_alpha_i)

            cnt += 150
            hidden = tuple(h.detach() for h in hidden)

        word_inputs = torch.stack(word_inputs, dim=1)
        char_outputs = torch.concat(char_outputs, dim=1)
        char_alpha = torch.concat(char_alpha, dim=1)

        word_outputs, _ = self.word_rnn(word_inputs)  # packed_word_inputs

        sentence_features, word_alpha = self.word_attention(word_outputs)

        char_features = char_outputs
        if seq_feat is not None:
            char_features = torch.concat([char_features, seq_feat.view(seq_feat.size(0), 1, seq_feat.size(1)).repeat(1,
                                                                                                                     char_outputs.size(
                                                                                                                         1),
                                                                                                                     1)],
                                         dim=2)
        if ser_feat is not None:
            char_features = torch.concat([char_features, ser_feat.view(ser_feat.size(0), 1, ser_feat.size(1)).repeat(1,
                                                                                                                     char_outputs.size(
                                                                                                                         1),
                                                                                                                     1)],
                                         dim=2)

        linear_outputs_1 = self.linear(char_features)
        linear_outputs = self.dropout_layer(linear_outputs_1)

        char_pred = self.head2(linear_outputs)
        seq_pred = self.head1(sentence_features)

        return char_pred ,seq_pred, seq_len, hidden, word_alpha, char_alpha

    def forward(self, input, seq_len, seq_feat=None, ser_feat=None):


        char_pred ,seq_pred, seq_length, hidden,word_alpha,char_alpha= self.logits(input,seq_len, None, None)

        return char_pred, seq_pred, seq_length,  hidden,word_alpha,char_alpha
