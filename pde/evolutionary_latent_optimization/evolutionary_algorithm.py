import torch
import pandas as pd
from tqdm import tqdm

from latent_dimension_assessment import sample_latent_vectors, latent_price_prediction_loss


class EvolutionaryAlgorithm:
    def __init__(
        self, 
        vae_trainer, 
        pinn_trainer, 
        latent_diagonal, 
        population_size, 
        mutation_strength, 
        selection_pressure_parameter, 
        n_generations,
        truncation_clip
    ):
        """
        Initialize the Evolutionary Algorithm class.

        Parameters:
        - vae_trainer: Instance of SurfaceVAETrainer (trained VAE).
        - pinn_trainer: Instance of DupirePINNTrainer (trained PINN).
        - latent_diagonal: Diagonal of the covariance matrix of the latent space prior.
        - population_size: Size of the population (number of latent vectors).
        - mutation_strength: Strength of the Gaussian mutation.
        - selection_pressure_parameter: Selection pressure parameter \( \eta \), controls the selection probability.
        - n_generations: Number of generations for the evolutionary optimization.
        - truncation_clip: Factor for clipping the latent vectors based on the latent prior's eigenvalues.
        """
        self.vae_trainer = vae_trainer
        self.pinn_trainer = pinn_trainer
        self.latent_diagonal = torch.tensor(latent_diagonal, dtype=torch.float32, device=self.vae_trainer.device)
        self.population_size = population_size
        self.mutation_strength = mutation_strength
        self.selection_pressure_parameter = selection_pressure_parameter
        self.n_generations = n_generations
        self.truncation_clip = truncation_clip

        # Initialize the population by sampling from the latent prior distribution
        self.population = sample_latent_vectors(self.population_size, latent_diagonal).to(self.vae_trainer.device)

        # Clip the population to stay within bounds
        self.clip_population()

        # Store optimization history (best/avg/0.05 percentile fitness at each generation)
        self.optimization_history = {
            "Best Fitness": [],
            "Average Fitness": [],
            "0.05 Percentile Fitness": []
        }

    def clip_population(self):
        """
        Clip the population of latent vectors to ensure they stay within the defined bounds.

        The bounds are determined by the truncation clip and the latent diagonal.
        """
        lower_bound = -self.truncation_clip * torch.sqrt(self.latent_diagonal)
        upper_bound = self.truncation_clip * torch.sqrt(self.latent_diagonal)

        # Clip the population to stay within the bounds
        self.population = torch.clamp(self.population, lower_bound, upper_bound)

    def mutate(self):
        """
        Mutate latent vectors by adding Gaussian noise based on mutation strength.
        Updates the population in place and applies clipping.
        """
        # Generate noise for mutation
        noise = self.mutation_strength * sample_latent_vectors(
            self.population_size,
            self.latent_diagonal.cpu().numpy()
        ).to(self.population.device)
        
        # Apply mutation by adding noise to the population
        self.population = self.population + noise

        # Clip the population to stay within bounds
        self.clip_population()

    def crossover(self):
        """
        Perform crossover operation to generate offspring by convex combination of random pairs of latent vectors.
        The operation is performed on the mutated population.
        Updates the population in place with offspring.
        """
        # Shuffle latent vectors randomly
        shuffled_indices = torch.randperm(self.population_size)
        shuffled_population = self.population[shuffled_indices]

        # Split into pairs for crossover
        pairs = shuffled_population.view(-1, 2, self.population.size(1))

        # Perform crossover for each pair
        alphas = torch.rand(pairs.size(0), 1).to(pairs.device)
        offspring = alphas * pairs[:, 0] + (1 - alphas) * pairs[:, 1]

        # Update the population with offspring
        self.population = torch.cat([self.population, offspring], dim=0)

    def selection(self):
        """
        Perform softmax-based selection for the next generation based on the fitness values.
        Updates the population in place by selecting based on softmax probabilities of fitness values.
        """
        # Evaluate fitness for the population
        fitness_values = self.fitness_evaluation()

        # Softmax selection based on fitness
        beta_n = (len(self.optimization_history["Average Fitness"]) + 1) ** self.selection_pressure_parameter
        probabilities = torch.softmax(beta_n * fitness_values, dim=0)
        
        selected_indices = torch.multinomial(probabilities, num_samples=self.population_size, replacement=True)
        
        self.population = self.population[selected_indices]

        return fitness_values[selected_indices]

    def fitness_evaluation(self):
        """
        Evaluate the fitness of the population using the VAE and PINN models.
        
        Returns:
        - fitness_values: Tensor of fitness values for the population.
        """
        # Compute price prediction losses for the latent vectors
        fitness_values = -latent_price_prediction_loss(self.population, self.vae_trainer, self.pinn_trainer)
        
        return fitness_values

    def optimize(self):
        """
        Perform the evolutionary optimization over the given number of generations.
        At the end of the optimization, a DataFrame of the final population is created,
        sorted by their price prediction loss (fitness).
        """

        for generation in tqdm(range(self.n_generations)):
            if generation > 0:
                # Mutation step
                self.mutate()

                # Crossover step (performed on mutated population)
                self.crossover()
                
                # Selection step
                fitness_values = self.selection()

            else:
                fitness_values = self.fitness_evaluation()

            # Track optimization history (best/avg/0.05 percentile fitness)
            best_fitness = torch.max(fitness_values).item()
            avg_fitness = torch.mean(fitness_values).item()
            percentile_fitness = torch.quantile(fitness_values, 0.05).item()
            
            self.optimization_history["Best Fitness"].append(best_fitness)
            self.optimization_history["Average Fitness"].append(avg_fitness)
            self.optimization_history["0.05 Percentile Fitness"].append(percentile_fitness)

            # Print the progress for each generation
            # print(f"Generation {generation + 1}/{self.n_generations}: Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness}, 0.05 Percentile Fitness: {percentile_fitness}")
        
        # Create the final population DataFrame sorted by price prediction loss
        final_price_prediction_loss = -self.fitness_evaluation().cpu().numpy()
        population_samples = self.vae_trainer.model.decode(self.population).detach().cpu().numpy()

        # Create the DataFrame with fitness as index
        final_population = pd.DataFrame(population_samples, index=final_price_prediction_loss)
        self.final_population = final_population.sort_index(ascending=True)  # Sort by price prediction loss (ascending)
