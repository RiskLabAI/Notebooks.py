from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from coefficients_vae_net import CoefficientsVAE, CoefficientsDataset, coefficients_beta_vae_loss
from torch.optim import Adam
import torch
from tqdm import tqdm


class SurfaceVAETrainer:
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        n_layers,
        data_dim,
        latent_diagonal,
        batch_size,
        beta,
        pre_train_learning_rate,
        fine_tune_learning_rate,
        pre_train_epochs,
        fine_tune_epochs,
        device,
    ):
        """
        Initialize the SurfaceVAETrainer class with the given hyperparameters and model configuration.
        """
        self.device = device
        self.latent_diagonal = torch.tensor(latent_diagonal, dtype=torch.float32, device=device)
        self.beta = beta
        self.batch_size = batch_size
        self.pre_train_learning_rate = pre_train_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.pre_train_epochs = pre_train_epochs
        self.fine_tune_epochs = fine_tune_epochs

        # Initialize the VAE model and optimizer
        self.model = CoefficientsVAE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            data_dim=data_dim,
        ).to(self.device)

        self.pre_train_optimizer = Adam(self.model.parameters(), lr=self.pre_train_learning_rate)
        self.fine_tune_optimizer = Adam(self.model.parameters(), lr=self.fine_tune_learning_rate)

    def _train(
        self, 
        sampled_surface_coefficients, 
        n_epochs, 
        optimizer, 
        loss_history, 
        experiment_name=None
    ):
        """
        Generic training function for the VAE model. It handles both pre-training and fine-tuning.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - n_epochs: The number of training epochs.
        - optimizer: The optimizer to use (Adam for pre-training or fine-tuning).
        - loss_history: A dictionary to keep track of the reconstruction and KL losses.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Create dataset and dataloader
        dataset = CoefficientsDataset(sampled_surface_coefficients)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Begin training
        for epoch in tqdm(range(n_epochs)):
            for batch_idx, batch_surface_coefficients in enumerate(dataloader):
                # Move data to the appropriate device
                batch_surface_coefficients = batch_surface_coefficients.to(self.device)

                # Zero gradients before backpropagation
                optimizer.zero_grad()

                # Forward pass through the VAE model
                self.model.train()
                reconstructed_surface_coefficients, latent_mean, latent_log_var = self.model(batch_surface_coefficients)

                # Compute the beta-VAE loss
                total_loss, reconstruction_loss, kl_divergence = coefficients_beta_vae_loss(
                    surface_coefficients=batch_surface_coefficients,
                    reconstructed_surface_coefficients=reconstructed_surface_coefficients,
                    latent_mean=latent_mean,
                    latent_log_var=latent_log_var,
                    latent_diagonal=self.latent_diagonal,
                    beta=self.beta
                )

                # Backpropagation and optimization
                total_loss.backward()
                optimizer.step()

                # Update loss dictionary
                loss_history["Reconstruction Loss"].append(reconstruction_loss.item())
                loss_history["KL Loss"].append(kl_divergence.item())
                loss_history["Total Loss"].append(total_loss.item())

                # # Current loss dict
                # current_loss = {
                #     "Reconstruction Loss": reconstruction_loss.item(),
                #     "KL Loss": kl_divergence.item(),
                #     "Total Loss": total_loss.item(),
                # }

                # # Print the losses for each batch
                # print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("Reconstruction Loss", reconstruction_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("KL Loss", kl_divergence.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Total Loss", total_loss.item(), epoch * len(dataloader) + batch_idx)

        # Close TensorBoard writer
        if writer:
            writer.close()

    def pre_train(
        self, 
        sampled_surface_coefficients, 
        experiment_name=None
    ):
        """
        Pre-train the VAE model.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        self._train(
            sampled_surface_coefficients,
            n_epochs=self.pre_train_epochs,
            optimizer=self.pre_train_optimizer,
            loss_history=self.pre_train_loss_history,
            experiment_name=experiment_name,
        )

    def fine_tune(
        self, 
        sampled_surface_coefficients, 
        experiment_name=None
    ):
        """
        Fine-tune the VAE model.

        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the Dataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.fine_tune_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        self._train(
            sampled_surface_coefficients,
            n_epochs=self.fine_tune_epochs,
            optimizer=self.fine_tune_optimizer,
            loss_history=self.fine_tune_loss_history,
            experiment_name=experiment_name,
        )

    def pre_train_with_sampling(
        self, 
        smoothness_prior, 
        experiment_name=None
    ):
        """
        Pre-train the VAE model with sampling from the smoothness prior.

        Parameters:
        - smoothness_prior: An instance of the RBFQuadraticSmoothnessPrior class used to sample surface coefficients.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "Reconstruction Loss": [],
            "KL Loss": [],
            "Total Loss": [],
        }

        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Begin training
        for epoch in tqdm(range(self.pre_train_epochs)):
            for batch_idx in range(1):
                # Sample surface coefficients from the smoothness prior
                sampled_surface_coefficients = smoothness_prior.sample_smooth_surfaces(self.batch_size)

                # Move sampled data to the appropriate device
                sampled_surface_coefficients = torch.tensor(sampled_surface_coefficients, device=self.device, dtype=torch.float32)

                # Zero gradients before backpropagation
                self.pre_train_optimizer.zero_grad()

                # Forward pass through the VAE model
                self.model.train()
                reconstructed_surface_coefficients, latent_mean, latent_log_var = self.model(sampled_surface_coefficients)

                # Compute the beta-VAE loss
                total_loss, reconstruction_loss, kl_divergence = coefficients_beta_vae_loss(
                    surface_coefficients=sampled_surface_coefficients,
                    reconstructed_surface_coefficients=reconstructed_surface_coefficients,
                    latent_mean=latent_mean,
                    latent_log_var=latent_log_var,
                    latent_diagonal=self.latent_diagonal,
                    beta=self.beta
                )

                # Backpropagation and optimization
                total_loss.backward()
                self.pre_train_optimizer.step()

                # Update loss dictionary
                self.pre_train_loss_history["Reconstruction Loss"].append(reconstruction_loss.item())
                self.pre_train_loss_history["KL Loss"].append(kl_divergence.item())
                self.pre_train_loss_history["Total Loss"].append(total_loss.item())

                # Print the losses for each batch
                # current_loss = {
                #     "Reconstruction Loss": reconstruction_loss.item(),
                #     "KL Loss": kl_divergence.item(),
                #     "Total Loss": total_loss.item(),
                # }
                # print(f"Epoch {epoch + 1}/{self.pre_train_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("Reconstruction Loss", reconstruction_loss.item(), epoch * self.batch_size + batch_idx)
                    writer.add_scalar("KL Loss", kl_divergence.item(), epoch * self.batch_size + batch_idx)
                    writer.add_scalar("Total Loss", total_loss.item(), epoch * self.batch_size + batch_idx)

        # Close TensorBoard writer
        if writer:
            writer.close()

    def save_model(
        self, 
        path='models/vae_model.pth'
    ):
        """
        Save the neural network model to a specified file path.

        Parameters:
        - path: The file path to save the model (including the file name, e.g., "model.pth").
        """
        torch.save(self.model.state_dict(), path)
        print(f"VAE Model saved to {path}.")

    def load_model(
        self, 
        path='models/vae_model.pth'
    ):
        """
        Load the neural network model from a specified file path.

        Parameters:
        - path: The file path from which to load the model (e.g., "model.pth").
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)  # Ensure the model is moved to the correct device after loading
        print(f"VAE Model loaded from {path}.")
