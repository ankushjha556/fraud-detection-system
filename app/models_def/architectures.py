"""
Model architecture definitions — must exactly mirror the training notebooks
(Sections 4 and 6) so saved state_dicts load correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# VAE — matches notebooks/03_vae.ipynb Cell 22
# ─────────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, latent_dim=8):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, 32)
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)
        self.dec1 = nn.Linear(latent_dim, 32)
        self.dec2 = nn.Linear(32, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ─────────────────────────────────────────────────────────────────────────
# LSTM — matches notebooks/05_lstm.ipynb Cell 36
# ─────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        h = F.relu(self.fc1(last_out))
        h = self.dropout(h)
        return self.fc2(h)


SEQ_LEN = 10  # must match Section 6 training config
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
