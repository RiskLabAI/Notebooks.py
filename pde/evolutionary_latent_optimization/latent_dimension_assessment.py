import torch

def sample_latent_vectors(
    n_samples, 
    latent_diagonal
):
    """
    Sample latent vectors from a zero-centered multivariate Gaussian distribution
    with a diagonal covariance matrix specified by latent_diagonal.

    Parameters:
    - n_samples: Number of samples to generate.
    - latent_diagonal: A 1D numpy array specifying the diagonal of the latent prior covariance matrix.

    Returns:
    - latent_samples: A tensor of shape (n_samples, latent_dim), where latent_dim is the length of latent_diagonal.
    """
    # Convert latent_diagonal to a PyTorch tensor
    latent_diagonal_tensor = torch.tensor(latent_diagonal, dtype=torch.float32)

    # Sample from a standard normal distribution and scale by the standard deviations (sqrt of the diagonal elements)
    latent_dim = latent_diagonal_tensor.size(0)
    standard_normal_samples = torch.randn(n_samples, latent_dim)

    # Scale the standard normal samples by the square root of the diagonal elements (i.e., standard deviations)
    latent_samples = standard_normal_samples * torch.sqrt(latent_diagonal_tensor)

    return latent_samples    


def latent_price_prediction_loss(
    latent_samples_batch,  # (batch, latent_dim)
    vae_trainer,           # An instance of SurfaceVAETrainer (with a trained VAE)
    pinn_trainer           # An instance of DupirePINNTrainer (with a trained PINN)
):
    """
    Calculates the price prediction loss for a batch of latent vectors by first decoding them
    into surface coefficients using the VAE and then passing the surface coefficients to the 
    dupire_price_prediction_loss function from the PINN trainer.

    Parameters:
    - latent_samples_batch: Tensor of shape (batch_size, latent_dim), representing the latent vectors.
    - vae_trainer: The instance of the trained SurfaceVAETrainer.
    - pinn_trainer: The instance of the trained DupirePINNTrainer.

    Returns:
    - price_prediction_errors_batch: Tensor of price prediction errors of shape (batch_size,).
    """

    # Move the latent samples to the correct device
    latent_samples_batch = latent_samples_batch.to(vae_trainer.device)

    # Set the VAE model to evaluation mode
    vae_trainer.model.eval()

    # Disable gradient tracking for model parameters
    requires_grad_backup = [param.requires_grad for param in vae_trainer.model.parameters()]
    for param in vae_trainer.model.parameters():
        param.requires_grad = False

    # Decode the latent vectors into surface coefficients
    decoded_surface_coefficients_batch = vae_trainer.model.decode(latent_samples_batch)

    # Restore the original requires_grad state for model parameters
    for param, requires_grad in zip(vae_trainer.model.parameters(), requires_grad_backup):
        param.requires_grad = requires_grad
    
    # Compute the price prediction loss using the surface coefficients batch
    price_prediction_errors_batch = pinn_trainer.dupire_price_prediction_loss(
        surface_coefficients_batch=decoded_surface_coefficients_batch
    )
    
    return price_prediction_errors_batch


def latent_space_assessment(
    latent_samples_batch,  # (batch, latent_dim)
    vae_trainer,           # An instance of SurfaceVAETrainer (with a trained VAE)
    pinn_trainer           # An instance of DupirePINNTrainer (with a trained PINN)
):
    """
    Efficiently performs an assessment of the latent space by calculating the condition number of the Hessian
    for each sample and estimating the Lipschitz constant by calculating the supremum of the 
    gradient norms of the price prediction error.

    Parameters:
    - latent_samples_batch: Tensor of shape (batch_size, latent_dim), representing the latent vectors.
    - vae_trainer: The instance of the trained SurfaceVAETrainer.
    - pinn_trainer: The instance of the trained DupirePINNTrainer.

    Returns:
    - condition_numbers: List of condition numbers of the Hessians for each latent sample.
    - lipschitz_constant: The estimated Lipschitz constant (supremum of gradient norms).
    """

    # Set requires_grad=True for each sample in the latent_samples_batch
    latent_samples_batch.requires_grad_(True)

    # Compute the price prediction errors for the entire batch
    price_prediction_errors_batch = latent_price_prediction_loss(
        latent_samples_batch=latent_samples_batch,
        vae_trainer=vae_trainer,
        pinn_trainer=pinn_trainer
    )

    # Sum the price prediction errors to compute the gradients efficiently
    total_price_prediction_error = price_prediction_errors_batch.sum()

    # Compute the gradients of the total error wrt the latent_samples_batch
    gradients_batch = torch.autograd.grad(
        total_price_prediction_error,
        latent_samples_batch,
        create_graph=True
    )[0]  # Shape: (batch_size, latent_dim)

    # Compute the norm of the gradients for Lipschitz constant estimation
    gradient_norms_batch = gradients_batch.norm(dim=1)  # Norm along the latent_dim dimension
    lipschitz_constant = gradient_norms_batch.max().item()  # Lipschitz constant is the maximum gradient norm

    # Initialize a placeholder for the batch of Hessians
    batch_size, latent_dim = latent_samples_batch.size()
    hessians_batch = torch.zeros(batch_size, latent_dim, latent_dim, device=vae_trainer.device)

    # Sum the gradients across the batch dimension for efficiency
    gradients_sum = gradients_batch.sum(dim=0)  # Shape: (latent_dim,)

    # Compute the Hessians for each sample
    for i in range(latent_dim):
        # Compute the second derivatives for the i-th dimension of the latent samples
        hessian_i = torch.autograd.grad(
            gradients_sum[i], 
            latent_samples_batch, 
            create_graph=True
        )[0]  # Shape: (batch_size, latent_dim)

        # Each row of hessian_i corresponds to the i-th row of the Hessian matrix for that sample
        hessians_batch[:, i, :] = hessian_i

    # Calculate the condition number of the Hessians for each sample in the batch
    condition_numbers = []
    for hessian in hessians_batch:
        # Symmetrize the Hessian before calculating eigenvalues
        hessian_sym = (hessian + hessian.T) / 2
        eigenvalues = torch.linalg.eigvalsh(hessian_sym)
        condition_number = eigenvalues.abs().max() / eigenvalues.abs().min()
        condition_numbers.append(condition_number.item())

    return condition_numbers, lipschitz_constant