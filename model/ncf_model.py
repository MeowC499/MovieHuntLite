import torch
import torch.nn as nn

class NCF_Hybrid(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=100):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=-1)
        return self.fc(x).squeeze()