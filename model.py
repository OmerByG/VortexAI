import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, words, vocab):
        self.words = words
        self.vocab = vocab

    def __len__(self):
        return len(self.words) - 1

    def __getitem__(self, idx):
        x = self.vocab[self.words[idx]]
        y = self.vocab[self.words[idx + 1]]
        return torch.tensor(x), torch.tensor(y)


class VortexModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        x = self.dropout(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden