import torch
import torch.nn as nn

class TimeDoseEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        if embedding_dim % 4 != 0:
            raise ValueError("Sinusoidal positional embedding cannot apply token dimension that is not a multiple of 4 (got dim={:d})".format(embedding_dim))
        self.embedding_dim = embedding_dim

    def forward(self, timestep, dose, max_period=1000):
        # timestep: shape (B,)
        # dose: shape (B,)
        device = timestep.device
        B = timestep.shape[0]
        d = self.embedding_dim // 2

        embeddings_t = torch.zeros(B, d).to(device)
        embeddings_d = torch.zeros(B, d).to(device)

        angle_rates = 1 / torch.pow(max_period, (2 * torch.arange(0, d//2).to(device) / d))

        embeddings_t[:, 0::2] = torch.sin(timestep.unsqueeze(1)/angle_rates)
        embeddings_t[:, 1::2] = torch.cos(timestep.unsqueeze(1)/angle_rates)

        embeddings_d[:, 0::2] = torch.sin(dose.unsqueeze(1)/angle_rates)
        embeddings_d[:, 1::2] = torch.cos(dose.unsqueeze(1)/angle_rates)

        embeddings = torch.cat((embeddings_t, embeddings_d), axis=1)

        return embeddings
