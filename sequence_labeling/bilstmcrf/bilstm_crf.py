import torch
from sequence_labeling.bilstmcrf.crf import CRF

class BiLSTMCRF(torch.nn.Module):
    def __init__(self, embedding, parameters, padding_idx, output_dim, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(BiLSTMCRF, self).__init__()
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
        num_tags = len(parameters['data_helper'].target2idx)
        start_idx = parameters['data_helper'].target2idx['[CLS]']
        self.crf = CRF(hidden_dim * 2, num_tags, start_idx=start_idx)

    def loss(self, sents, tags):
        x = self.embedding(sents)
        mask = sents.gt(0)
        outputs, (hidden, cell) = self.lstm(x)
        features = self.dropout(outputs)
        loss = self.crf.loss(features, tags, mask)
        return loss

    def forward(self, sents):
        x = self.embedding(sents)
        mask = sents.gt(0)
        outputs, (hidden, cell) = self.lstm(x)
        features = self.dropout(outputs)
        scores, tag_seq = self.crf(features, mask)
        return scores, tag_seq
