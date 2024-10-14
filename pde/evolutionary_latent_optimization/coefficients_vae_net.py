import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CoefficientsVAE(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        hidden_dim, 
        n_layers, 
        data_dim
    ):
        """
        Initialize the CoefficientsVAE class.

        Parameters:
        - latent_dim: Dimension of the latent space z.
        - hidden_dim: Dimension of the hidden layers in the encoder and decoder.
        - n_layers: Number of hidden layers in both encoder and decoder.
        - data_dim: Dimension of the input surface coefficients ω.
        """
        super(CoefficientsVAE, self).__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim

        # Define the encoder network
        encoder_layers = []
        input_dim = data_dim
        for _ in range(n_layers):
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ELU())
            input_dim = hidden_dim

        # Output layers for mean and log variance
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

        # Define the decoder network
        decoder_layers = []
        input_dim = latent_dim
        for _ in range(n_layers):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ELU())
            input_dim = hidden_dim

        # Output layer for the reconstructed surface coefficients
        self.decoder = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(hidden_dim, data_dim)

        # Apply Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Apply Xavier normal initialization to weights and set biases to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self, 
        surface_coefficients
    ):
        """
        Encode surface coefficients ω into latent variables z.

        Parameters:
        - surface_coefficients: Input surface coefficients ω.

        Returns:
        - mu: Mean of the latent distribution q(z|ω).
        - log_var: Log variance of the latent distribution q(z|ω).
        """
        encoded = self.encoder(surface_coefficients)
        mu = self.mu_layer(encoded)
        log_var = self.log_var_layer(encoded)

        return mu, log_var

    def decode(
        self, 
        latent_sample
    ):
        """
        Decode latent variable z into surface coefficients ω.

        Parameters:
        - latent_sample: Latent variables z.

        Returns:
        - reconstructed_surface_coefficients: Reconstructed surface coefficients ω.
        """
        decoded = self.decoder(latent_sample)
        reconstructed_surface_coefficients = self.output_layer(decoded)

        return reconstructed_surface_coefficients

    def forward(
        self, 
        surface_coefficients
    ):
        """
        Forward pass through the VAE. Encodes the surface coefficients, samples from the latent space,
        and reconstructs the surface coefficients.

        Parameters:
        - surface_coefficients: Input surface coefficients ω.

        Returns:
        - reconstructed_surface_coefficients: Reconstructed surface coefficients ω.
        - mu: Mean of the latent distribution q(z|ω).
        - log_var: Log variance of the latent distribution q(z|ω).
        """
        # Encode the surface coefficients to get mu and log_var
        mu, log_var = self.encode(surface_coefficients)

        # Reparameterization trick to sample z
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Standard normal noise
        latent_sample = mu + eps * std  # Reparameterization

        # Decode the latent sample to reconstruct the surface coefficients
        reconstructed_surface_coefficients = self.decode(latent_sample)

        return reconstructed_surface_coefficients, mu, log_var


def coefficients_beta_vae_loss(
    surface_coefficients, 
    reconstructed_surface_coefficients, 
    latent_mean, 
    latent_log_var, 
    latent_diagonal, 
    beta
):
        """
        Compute the beta-VAE loss, including the reconstruction loss and the KL divergence with a structured prior.

        Parameters:
        - surface_coefficients: Ground truth surface coefficients ω (batch_size, data_dim).
        - reconstructed_surface_coefficients: Reconstructed surface coefficients from the VAE decoder (batch_size, data_dim).
        - latent_mean: Mean of the latent distribution q(z|ω) (batch_size, latent_dim).
        - latent_log_var: Log variance of the latent distribution q(z|ω) (batch_size, latent_dim).
        - latent_diagonal: Diagonal matrix for the latent prior covariance structure (latent_dim).
        - beta: Regularization parameter for the KL divergence term.

        Returns:
        - total_loss: The total beta-VAE loss (reconstruction loss + β * KL divergence) averaged over the batch.
        - reconstruction_loss: The reconstruction loss component of the total loss, averaged over the batch.
        - kl_divergence: The KL divergence component of the total loss, averaged over the batch.
        """

        # Compute the reconstruction loss (mean squared error between true and reconstructed coefficients)
        # Reduction is 'none' so we compute MSE per batch sample and then average
        reconstruction_loss = F.mse_loss(reconstructed_surface_coefficients, surface_coefficients, reduction='none')
        reconstruction_loss = reconstruction_loss.sum(dim=1)  # Compute the mean MSE per batch sample
        reconstruction_loss = reconstruction_loss.mean()  # Average across the batch

        # Compute the KL divergence between q(z|ω) and p(z)
        latent_var = torch.exp(latent_log_var)  # Convert log variance to variance

        # Compute KL divergence for each latent dimension and each batch sample
        kl_divergence = 0.5 * torch.sum(
            (latent_var / latent_diagonal) + (latent_mean ** 2 / latent_diagonal) - 1 - latent_log_var + torch.log(latent_diagonal),
            dim=1
        )

        # Average KL divergence across the batch
        kl_divergence = kl_divergence.mean()

        # Compute the total loss as the sum of the reconstruction and KL divergence, weighted by beta
        total_loss = reconstruction_loss + beta * kl_divergence

        return total_loss, reconstruction_loss, kl_divergence


class CoefficientsDataset(Dataset):
    def __init__(self, sampled_surface_coefficients):
        """
        Initialize the dataset with sampled surface coefficients.

        Parameters:
        - sampled_surface_coefficients: 2D NumPy array of surface coefficients (n_samples, n_coefficients).
        """
        self.data = torch.tensor(sampled_surface_coefficients, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the coefficients for a given index as a tensor
        return self.data[idx]        