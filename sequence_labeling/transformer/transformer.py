import math
import torch

class Transformer(torch.nn.Module):
    def __init__(self, parameters, padding_idx, output_dim, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        num_embeddings = len(parameters['tokenizer'].vocab)
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(num_embeddings, self.d_model)
        self.embedding.padding_idx = padding_idx
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = torch.nn.LayerNorm(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.output2label = torch.nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output2label.bias.data.zero_()
        self.output2label.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, sents):
        if self.src_mask is None or self.src_mask.size(0) != len(sents):
            device = sents.device
            mask = self._generate_square_subsequent_mask(len(sents)).to(device)
            self.src_mask = mask
        src = self.embedding(sents) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, self.src_mask)
        output = self.output2label(output)
        output = output.transpose(1, 2)
        # CrossEntropyLoss input(predict vector) shape is (minibatch,C) or (minibatch, C, d_1, d_2, ..., d_K)
        # C = num classes
        return output

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)