import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):

        att1 = self.encoder_att(features)
        att2 = self.decoder_att(hidden_state).unsqueeze(1)

        att = torch.tanh(att1 + att2)
        scores = self.full_att(att)
        alphas = F.softmax(scores, dim=1)
        context_vector = features * alphas
        context_vector = context_vector.sum(dim=1)

        return context_vector, alphas
