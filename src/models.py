import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    """
    Vanilla single-layer RNN model.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        out, _ = self.rnn(embedded)   # shape: (batch_size, seq_len, hidden_dim)
        logits = self.fc(out)         # shape: (batch_size, seq_len, num_classes)
        return logits

class LSTMModel(nn.Module):
    """
    Single-layer LSTM model.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out)
        return logits

class GRUModel(nn.Module):
    """
    Single-layer GRU model.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        logits = self.fc(out)
        return logits

class BidirectionalModel(nn.Module):
    """
    Bidirectional LSTM model.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out)
        return logits

class DeepRNNModel(nn.Module):
    """
    Stacked LSTM model (2 layers).
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        logits = self.fc(out)
        return logits

class BidirectionalGRUModel(nn.Module):
    """
    Bidirectional GRU model.
    """
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.gru(embedded)
        logits = self.fc(out)
        return logits

