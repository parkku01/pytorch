import torch

class BiLSTM(torch.nn.Module):
    def __init__(self, embedding, parameters, padding_idx, output_dim, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(BiLSTM, self).__init__()
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
        self.hidden2label = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, sents):
        x = self.embedding(sents)
        outputs, (hidden, cell) = self.lstm(x)
        predictions = self.hidden2label(self.dropout(outputs))
        predictions = predictions.transpose(1, 2)
        # CrossEntropyLoss input(predict vector) shape is (minibatch,C) or (minibatch, C, d_1, d_2, ..., d_K)
        # C = num classes
        return predictions