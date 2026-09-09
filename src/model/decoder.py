import torch
import torch.nn as nn
from src.model.attention import Attention


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, decoder_dim, attention_dim, vocab_size, encoder_dim=2048, drop_prob=0.5):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim,
                                decoder_dim, bias=True)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        batch_size = features.size(0)
        seq_length = captions.size(1) - 1
        mean_features = features.mean(dim=1)

        h = self.init_h(mean_features)
        c = self.init_c(mean_features)

        predictions = torch.zeros(
            batch_size, seq_length, self.vocab_size).to(features.device)
        alphas = torch.zeros(batch_size, seq_length,
                             features.size(1)).to(features.device)

        for t in range(seq_length):
            context_vector, alpha = self.attention(features, h)
            current_embedding = embeddings[:, t, :]
            lstm_input = torch.cat((current_embedding, context_vector), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            predictions[:, t, :] = output
            alphas[:, t, :] = alpha.squeeze(2)

        return predictions, alphas
