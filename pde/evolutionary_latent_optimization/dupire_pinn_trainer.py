import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from call_option_net import CallOptionPINN, pinn_dupire_loss
from rbf_volatility_surface import SurfaceDataset
from tqdm import tqdm
from itertools import product

class DupirePINNTrainer:
    def __init__(
        self,
        hidden_dim,
        n_layers,
        batch_size,
        pde_loss_coefficient,
        maturity_zero_loss_coefficient,
        strike_zero_loss_coefficient,
        strike_infinity_loss_coefficient,
        pre_train_learning_rate,
        fine_tune_learning_rate,
        pre_train_epochs,
        fine_tune_epochs,
        maturity_min,
        maturity_max,
        strike_min,
        strike_max,
        volatility_mean,
        volatility_std,
        maturity_time_list,
        strike_price_list,
        strike_std,
        maturity_std,
        constant_volatility,
        strike_infinity,
        device,
    ):
        """
        Initialize the DupirePINNTrainer class with the given hyperparameters and model configuration.
        """
        self.device = device
        self.batch_size = batch_size
        self.pde_loss_coefficient = pde_loss_coefficient
        self.maturity_zero_loss_coefficient = maturity_zero_loss_coefficient
        self.strike_zero_loss_coefficient = strike_zero_loss_coefficient
        self.strike_infinity_loss_coefficient = strike_infinity_loss_coefficient
        self.pre_train_learning_rate = pre_train_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.pre_train_epochs = pre_train_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.maturity_time_list = maturity_time_list
        self.strike_price_list = strike_price_list
        self.strike_std = strike_std
        self.maturity_std = maturity_std
        self.constant_volatility = constant_volatility
        self.strike_infinity = strike_infinity

        # Initialize the PINN model and optimizer
        self.model = CallOptionPINN(
            hidden_dim,
            n_layers,
            maturity_min,
            maturity_max,
            strike_min,
            strike_max,
            volatility_mean,
            volatility_std,
        ).to(self.device)

        self.pre_train_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pre_train_learning_rate)
        self.fine_tune_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.fine_tune_learning_rate)

    def _train(
        self, 
        sampled_surface_coefficients, 
        n_epochs, 
        optimizer, 
        loss_history, 
        experiment_name=None
    ):
        """
        Generic training function for the PINN model. It handles both pre-training and fine-tuning.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - n_epochs: The number of training epochs.
        - optimizer: The optimizer to use (Adam for pre-training or fine-tuning).
        - loss_history: A dictionary to keep track of the individual losses and the total loss.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Create dataset and dataloader
        dataset = SurfaceDataset(
            sampled_surface_coefficients=sampled_surface_coefficients,
            maturity_time_list=self.maturity_time_list,
            strike_price_list=self.strike_price_list,
            strike_std=self.strike_std,
            maturity_std=self.maturity_std,
            constant_volatility=self.constant_volatility,
            strike_infinity=self.strike_infinity,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Begin training
        for epoch in tqdm(range(n_epochs)):
            for batch_idx, (time_to_maturity, strike_price, implied_volatility) in enumerate(dataloader):
                # Move data to the appropriate device
                time_to_maturity = time_to_maturity.to(self.device)
                strike_price = strike_price.to(self.device)
                implied_volatility = implied_volatility.to(self.device)

                # Zero gradients before backpropagation
                optimizer.zero_grad()

                # Forward pass through the model to get call option price predictions
                self.model.train()
                call_option_price = self.model(time_to_maturity, strike_price, implied_volatility)

                # Compute the PDE and boundary condition losses
                pde_loss, maturity_zero_loss, strike_zero_loss, strike_infinity_loss = pinn_dupire_loss(
                    call_option_price,
                    time_to_maturity,
                    strike_price,
                    implied_volatility,
                    strike_infinity=self.strike_infinity,
                )

                # Compute total loss with coefficients
                total_loss = (
                    self.pde_loss_coefficient * pde_loss
                    + self.maturity_zero_loss_coefficient * maturity_zero_loss
                    + self.strike_zero_loss_coefficient * strike_zero_loss
                    + self.strike_infinity_loss_coefficient * strike_infinity_loss
                )

                # Backpropagation and optimization
                total_loss.backward()
                optimizer.step()

                # Update loss dictionary
                loss_history["PDE Loss"].append(pde_loss.item())
                loss_history["Zero Maturity Loss"].append(maturity_zero_loss.item())
                loss_history["Zero Strike Loss"].append(strike_zero_loss.item())
                loss_history["Infinity Strike Loss"].append(strike_infinity_loss.item())
                loss_history["Total Loss"].append(total_loss.item())

                # current_loss = {
                #     "PDE Loss": pde_loss.item(),
                #     "Zero Maturity Loss": maturity_zero_loss.item(),
                #     "Zero Strike Loss": strike_zero_loss.item(),
                #     "Infinity Strike Loss": strike_infinity_loss.item(),
                #     "Total Loss": total_loss.item(),
                # }

                # # Print the losses for each batch
                # print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("PDE Loss", pde_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Zero Maturity Loss", maturity_zero_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Zero Strike Loss", strike_zero_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar("Infinity Strike Loss", strike_infinity_loss.item(), epoch * len(dataloader) + batch_idx)
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
        Pre-train the PINN model.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "PDE Loss": [],
            "Zero Maturity Loss": [],
            "Zero Strike Loss": [],
            "Infinity Strike Loss": [],
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
        Fine-tune the PINN model.
        
        Parameters:
        - sampled_surface_coefficients: The sampled surface coefficients used to create the SurfaceDataset.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.fine_tune_loss_history = {
            "PDE Loss": [],
            "Zero Maturity Loss": [],
            "Zero Strike Loss": [],
            "Infinity Strike Loss": [],
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
        Pre-train the PINN model with sampling from the smoothness prior.

        Parameters:
        - smoothness_prior: An instance of the RBFQuadraticSmoothnessPrior class used to sample surface coefficients.
        - experiment_name: If provided, logs will be recorded in TensorBoard with this experiment name.
        """
        self.pre_train_loss_history = {
            "PDE Loss": [],
            "Zero Maturity Loss": [],
            "Zero Strike Loss": [],
            "Infinity Strike Loss": [],
            "Total Loss": [],
        }

        # Setup SummaryWriter for TensorBoard logging
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}") if experiment_name else None

        # Begin pre-training with sampling at each epoch
        for epoch in tqdm(range(self.pre_train_epochs)):
            # Sample surface coefficients from the smoothness prior for this epoch
            sampled_surface_coefficients = smoothness_prior.sample_smooth_surfaces(self.batch_size)

            # Create a dataset from the sampled surface coefficients
            dataset = SurfaceDataset(
                sampled_surface_coefficients=sampled_surface_coefficients,
                maturity_time_list=self.maturity_time_list,
                strike_price_list=self.strike_price_list,
                strike_std=self.strike_std,
                maturity_std=self.maturity_std,
                constant_volatility=self.constant_volatility,
                strike_infinity=self.strike_infinity,
            )

            # Take the entire dataset (no batching) for this epoch
            dataloader = DataLoader(dataset, batch_size=self.batch_size)

            # Iterate through the entire dataset (no batching, only one batch)
            for batch_idx, (time_to_maturity, strike_price, implied_volatility) in enumerate(dataloader):
                # Move data to the appropriate device
                time_to_maturity = time_to_maturity.to(self.device)
                strike_price = strike_price.to(self.device)
                implied_volatility = implied_volatility.to(self.device)

                # Zero gradients before backpropagation
                self.pre_train_optimizer.zero_grad()

                # Forward pass through the model to get call option price predictions
                self.model.train()
                call_option_price = self.model(time_to_maturity, strike_price, implied_volatility)

                # Compute the PDE and boundary condition losses
                pde_loss, maturity_zero_loss, strike_zero_loss, strike_infinity_loss = pinn_dupire_loss(
                    call_option_price,
                    time_to_maturity,
                    strike_price,
                    implied_volatility,
                    strike_infinity=self.strike_infinity,
                )

                # Compute total loss with coefficients
                total_loss = (
                    self.pde_loss_coefficient * pde_loss
                    + self.maturity_zero_loss_coefficient * maturity_zero_loss
                    + self.strike_zero_loss_coefficient * strike_zero_loss
                    + self.strike_infinity_loss_coefficient * strike_infinity_loss
                )

                # Backpropagation and optimization
                total_loss.backward()
                self.pre_train_optimizer.step()

                # Update loss dictionary
                self.pre_train_loss_history["PDE Loss"].append(pde_loss.item())
                self.pre_train_loss_history["Zero Maturity Loss"].append(maturity_zero_loss.item())
                self.pre_train_loss_history["Zero Strike Loss"].append(strike_zero_loss.item())
                self.pre_train_loss_history["Infinity Strike Loss"].append(strike_infinity_loss.item())
                self.pre_train_loss_history["Total Loss"].append(total_loss.item())

                # # Print the losses for this epoch
                # current_loss = {
                #     "PDE Loss": pde_loss.item(),
                #     "Zero Maturity Loss": maturity_zero_loss.item(),
                #     "Zero Strike Loss": strike_zero_loss.item(),
                #     "Infinity Strike Loss": strike_infinity_loss.item(),
                #     "Total Loss": total_loss.item(),
                # }

                # # Print the losses
                # print(f"Epoch {epoch + 1}/{self.pre_train_epochs}, Losses: {current_loss}")

                # Log losses in TensorBoard
                if writer:
                    writer.add_scalar("PDE Loss", pde_loss.item(), epoch)
                    writer.add_scalar("Zero Maturity Loss", maturity_zero_loss.item(), epoch)
                    writer.add_scalar("Zero Strike Loss", strike_zero_loss.item(), epoch)
                    writer.add_scalar("Infinity Strike Loss", strike_infinity_loss.item(), epoch)
                    writer.add_scalar("Total Loss", total_loss.item(), epoch)

        # Close TensorBoard writer
        if writer:
            writer.close()
    
    def save_model(
        self, 
        path='models/pinn_model.pth'
    ):
        """
        Save the neural network model to a specified file path.

        Parameters:
        - path: The file path to save the model (including the file name, e.g., "model.pth").
        """
        torch.save(self.model.state_dict(), path)
        print(f"PINN Model saved to {path}.")

    def load_model(
        self, 
        path='models/pinn_model.pth'
    ):
        """
        Load the neural network model from a specified file path.

        Parameters:
        - path: The file path from which to load the model (e.g., "model.pth").
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)  # Ensure the model is moved to the correct device after loading
        print(f"PINN Model loaded from {path}.")

    def dupire_price_prediction_loss(
        self,
        surface_coefficients_batch,
        data_call_option_prices=None,
        data_maturity_times=None,
        data_strike_prices=None
    ):
        """
        Calculate the price prediction loss for a batch of surface coefficients.

        Parameters:
        - surface_coefficients_batch: A batch of surface coefficients with shape (batch, N).
        - data_call_option_prices: Observed call option prices. If provided, set the corresponding attribute.
        - data_maturity_times: Maturity times corresponding to the observed call option prices.
        - data_strike_prices: Strike prices corresponding to the observed call option prices.

        Returns:
        - mse_loss: The mean squared error (MSE) loss between the predicted and observed call option prices.
        """

        # Set class attributes if provided
        if data_call_option_prices is not None:
            self.data_call_option_prices = torch.tensor(data_call_option_prices, dtype=torch.float32, device=self.device)
        if data_maturity_times is not None:
            self.data_maturity_times = torch.tensor(data_maturity_times, dtype=torch.float32, device=self.device)
        if data_strike_prices is not None:
            self.data_strike_prices = torch.tensor(data_strike_prices, dtype=torch.float32, device=self.device)

        # Ensure that RBF evaluations are computed if not already cached
        if not hasattr(self, 'rbf_evaluations') or self.rbf_evaluations is None:
            pde_grid = list(product(self.maturity_time_list, self.strike_price_list))
            maturity_times, strike_prices = zip(*pde_grid)

            # Expand dimensions to enable broadcasting and compute RBF evaluations
            time_diff = (self.data_maturity_times[:, None] - torch.tensor(maturity_times, device=self.device)) ** 2
            strike_diff = (self.data_strike_prices[:, None] - torch.tensor(strike_prices, device=self.device)) ** 2

            # rbf_evaluations: (M, N)
            self.rbf_evaluations = torch.exp(
                -time_diff / (2 * self.maturity_std ** 2)
                - strike_diff / (2 * self.strike_std ** 2)
            ).float()

        # surface_coefficients_batch: (batch_size, N)

        # Compute the predicted volatilities for each surface coefficients batch
        predicted_volatility_batch = torch.matmul(surface_coefficients_batch, self.rbf_evaluations.T) + self.constant_volatility

        # Now predict the call option prices using the PINN for each batch element
        # Repeat the maturity and strike tensors across the batch dimension
        repeated_maturity_times = self.data_maturity_times.unsqueeze(0).repeat(surface_coefficients_batch.size(0), 1)
        repeated_strike_prices = self.data_strike_prices.unsqueeze(0).repeat(surface_coefficients_batch.size(0), 1)

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking for model parameters
        requires_grad_backup = [param.requires_grad for param in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad = False

        # Pass through the model
        predicted_call_option_prices = self.model(
            repeated_maturity_times,  # Shape: (batch_size, M)
            repeated_strike_prices,   # Shape: (batch_size, M)
            predicted_volatility_batch  # Shape: (batch_size, M)
        )

        # Restore the original requires_grad state for model parameters
        for param, requires_grad in zip(self.model.parameters(), requires_grad_backup):
            param.requires_grad = requires_grad

        # We now have predicted_call_option_prices of shape (batch_size, M)

        # Ensure that the observed prices are of the correct shape
        repeated_observed_prices = self.data_call_option_prices.unsqueeze(0).expand_as(predicted_call_option_prices)

        # Compute the squared differences between predicted and observed call option prices
        squared_errors = (predicted_call_option_prices - repeated_observed_prices) ** 2

        # Sum the squared errors over the M points (along the second dimension)
        sum_squared_errors_batch = torch.sum(squared_errors, dim=1)  # Shape: (batch_size,)

        return sum_squared_errors_batch       