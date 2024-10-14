import plotly.graph_objects as go

from evolutionary_algorithm import EvolutionaryAlgorithm

class AdaptiveEvolutionaryLatentOptimization:
    def __init__(
        self,
        vae_trainer,
        pinn_trainer,
        latent_diagonal,
        population_size,
        mutation_strength,
        selection_pressure_parameter,
        n_generations,
        truncation_clip,
        n_cycles,
    ):
        """
        Initialize the AdELO class with pre-trained models and hyperparameters.

        Parameters:
        - vae_trainer: Pre-trained VAE trainer instance.
        - pinn_trainer: Pre-trained PINN trainer instance.
        - latent_diagonal: Diagonal of the latent prior covariance matrix.
        - population_size: Population size for the EA.
        - mutation_strength: Mutation strength parameter for the EA.
        - selection_pressure_parameter: Selection pressure parameter for the EA.
        - n_generations: Number of generations per EA optimization cycle.
        - truncation_clip: Factor for clipping the latent vectors based on the latent prior's eigenvalues.
        - n_cycles: Number of adaptive cycles.
        """
        self.vae_trainer = vae_trainer
        self.pinn_trainer = pinn_trainer
        self.latent_diagonal = latent_diagonal
        self.population_size = population_size
        self.mutation_strength = mutation_strength
        self.selection_pressure_parameter = selection_pressure_parameter
        self.n_generations = n_generations
        self.truncation_clip = truncation_clip
        self.n_cycles = n_cycles

        # Histories for each cycle
        self.optimization_histories = []

        # Placeholder for final population after the last cycle
        self.final_population = None

    def run_cycle(self):
        """
        Runs the adaptive cycles of EA and fine-tuning.
        """
        for cycle in range(self.n_cycles):
            print(f"Starting Cycle {cycle + 1}/{self.n_cycles}")
            
            # Skip fine-tuning for the first cycle, perform fine-tuning for subsequent cycles
            if cycle > 0:
                print("Fine-Tuning VAE and PINN...")
                self.vae_trainer.fine_tune(self.final_population.values)
                self.pinn_trainer.fine_tune(self.final_population.values)

            # Run the evolutionary optimization (EA step)
            print("Running Evolutionary Optimization...")
            ea_optimizer = EvolutionaryAlgorithm(
                vae_trainer=self.vae_trainer,
                pinn_trainer=self.pinn_trainer,
                latent_diagonal=self.latent_diagonal,
                population_size=self.population_size,
                mutation_strength=self.mutation_strength,
                selection_pressure_parameter=self.selection_pressure_parameter,
                n_generations=self.n_generations,
                truncation_clip=self.truncation_clip
            )
            ea_optimizer.optimize()

            # Store the final population and optimization history
            self.optimization_histories.append(ea_optimizer.optimization_history)

            # Update the final population with the current EA optimizer's population
            self.final_population = ea_optimizer.final_population

            print(f"Cycle {cycle + 1}/{self.n_cycles} Completed.\n")

    def plot_evolutions(self):
        """
        Merges all the optimization histories from each cycle and plots them in a combined figure.
        """
        # Initialize figure
        fig = go.Figure()

        # Initialize lists to hold the combined data across all cycles
        merged_best_fitness = []
        merged_average_fitness = []
        merged_percentile_fitness = []
        generation_numbers = []

        # Track the total number of generations
        total_generations = 0

        # Loop through the optimization histories of all cycles
        for cycle_idx, optimization_history in enumerate(self.optimization_histories):
            # Number of generations in the current cycle
            n_generations_cycle = len(optimization_history["Best Fitness"])

            # Generate generation numbers for the current cycle
            generation_numbers_cycle = list(range(total_generations + 1, total_generations + n_generations_cycle + 1))

            # Append the data from the current cycle to the merged lists
            merged_best_fitness.extend(optimization_history["Best Fitness"])
            merged_average_fitness.extend(optimization_history["Average Fitness"])
            merged_percentile_fitness.extend(optimization_history["0.05 Percentile Fitness"])
            generation_numbers.extend(generation_numbers_cycle)

            # Update total generations
            total_generations += n_generations_cycle

        # Add trace for best fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_best_fitness,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='royalblue', width=3),
            marker=dict(size=6, symbol='circle'),
            hovertemplate='<b>Iteration %{x}</b><br>Best Fitness: %{y:.4f}<extra></extra>'
        ))

        # Add trace for average fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_average_fitness,
            mode='lines+markers',
            name='Average Fitness',
            line=dict(color='firebrick', width=3, dash='dash'),
            marker=dict(size=6, symbol='square'),
            hovertemplate='<b>Iteration %{x}</b><br>Average Fitness: %{y:.4f}<extra></extra>'
        ))

        # Add trace for 0.05 percentile fitness
        fig.add_trace(go.Scatter(
            x=generation_numbers,
            y=merged_percentile_fitness,
            mode='lines+markers',
            name='0.05 Percentile Fitness',
            line=dict(color='green', width=3, dash='dot'),
            marker=dict(size=6, symbol='triangle-up'),
            hovertemplate='<b>Iteration %{x}</b><br>0.05 Percentile Fitness: %{y:.4f}<extra></extra>'
        ))

        # Update layout to make the plot detailed and visually appealing
        fig.update_layout(
            title="Adaptive Latent Evolutionary Optimization History Across All Cycles",
            xaxis_title="Generation",
            yaxis_title="Fitness Value",
            legend=dict(
                x=0.75,
                y=0.15,
            ),
            hovermode='x unified',
            width=900,
            height=900,
        )

        # Show the plot
        fig.show()           

        fig.write_image('figs/adaptive_latent_evolutionary_optimization_history.png', format='png', scale=4, width=900, height=900) 
