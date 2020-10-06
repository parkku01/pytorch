import torch
import numpy as np
import torch.nn.functional as F

class BiLSTMATT(torch.nn.Module):
    def __init__(self, embedding, parameters, padding_idx, output_dim, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(BiLSTMATT, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(p=dropout)
        if embedding:
            self.embedding = torch.nn.Embedding.from_pretrained(embedding)
        else:
            num_embeddings = len(parameters['tokenizer'].vocab)
            embedding_dim = parameters.get('embedding_dim')
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = torch.nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.attention = Attention(hidden_dim * 2, batch_first=True)
        self.hidden2label = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sents, x_len):
        x = self.embedding(sents)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(x)
        pad_outputs, lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        att_outputs, _ = self.attention(pad_outputs, lengths)
        predictions = self.hidden2label(self.dropout(att_outputs))
        predictions = predictions.transpose(1, 2)
        # CrossEntropyLoss input(predict vector) shape is (minibatch,C) or (minibatch, C, d_1, d_2, ..., d_K)
        # C = num classes
        return predictions


class Attention(torch.nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = torch.nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # get the final fixed vector representations of the sentences
        # representations = weighted.sum(1).squeeze()

        return weighted, attentions