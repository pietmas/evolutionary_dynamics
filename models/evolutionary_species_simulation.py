import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display
import matplotlib.colors as mcolors
import time

class EvolutionarySpeciesSimulation:
    """
    Simulates multiple species interacting in an ecosystem on a 2D grid.
    Each species has its own birth and death rates, and interactions between species 
    are governed by an interaction matrix, where positive values represent beneficial 
    interactions and negative values represent harmful interactions.
    """

    def __init__(self, species_params, interaction_matrix,
                 grid_size=50, steps=200):
        """
        Initializes the simulation.

        Parameters:
        - species_params: A list of dictionaries containing parameters for each species.
          Each dictionary should have:
            - 'name': Name of the species
            - 'initial_population': Initial number of agents
            - 'birth_rate': Probability of reproducing each time step
            - 'death_rate': Probability of dying naturally each time step
            - 'color': Color for visualization
        - interaction_matrix: A 2D numpy array representing interaction coefficients between species.
          Positive values indicate beneficial interactions, negative values indicate harmful interactions.
        - grid_size: Size of the grid (grid_size x grid_size)
        - steps: Number of simulation steps
        """
        self.species_params = species_params
        self.interaction_matrix = interaction_matrix
        self.num_species = len(species_params)
        self.grid_size = grid_size
        self.steps = steps

        # Initialize grids and population counts
        self.species_grids = [
            np.zeros((grid_size, grid_size), dtype=int) for _ in range(self.num_species)
        ]
        self.population_counts = [[] for _ in range(self.num_species)]

        self.initialize_population()

    def initialize_population(self):
        """
        Places the initial population for each species randomly on the grid.
        """
        for idx, params in enumerate(self.species_params):
            for _ in range(params['initial_population']):
                x, y = np.random.randint(0, self.grid_size, size=2)
                self.species_grids[idx][x, y] += 1

            self.population_counts[idx].append(params['initial_population'])

    def move_agents(self, grid):
        """
        Moves each agent on the grid to a random neighboring cell.

        Parameters:
        - grid: The current grid of a species

        Returns:
        - new_grid: The updated grid after agents have moved
        """
        new_grid = np.zeros_like(grid)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                count = grid[x, y]
                for _ in range(count):
                    dx, dy = np.random.choice([-1, 0, 1], size=2)
                    new_x = (x + dx) % self.grid_size
                    new_y = (y + dy) % self.grid_size
                    new_grid[new_x, new_y] += 1
        return new_grid

    def simulate(self):
        """
        Runs the simulation for the specified number of steps.
        At each step, agents move, reproduce, die, and interact based on the interaction matrix.
        The system state is visualized at each step.
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]}
        )
        plt.subplots_adjust(wspace=0.4)

        img = np.zeros((self.grid_size, self.grid_size, 3))
        im = ax1.imshow(img)
        ax1.set_title("Initial State")
        ax1.axis('off')

        population_lines = []
        for idx, params in enumerate(self.species_params):
            line, = ax2.plot([], [], label=params['name'], color=params['color'])
            population_lines.append(line)
        ax2.set_xlim(0, self.steps)
        initial_max_population = max([params['initial_population'] for params in self.species_params]) * 1.2
        ax2.set_ylim(0, initial_max_population)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Population')
        ax2.set_title('Population Trends Over Time')
        ax2.legend()

        for step in range(self.steps):
            for idx in range(self.num_species):
                self.species_grids[idx] = self.move_agents(self.species_grids[idx])

            for idx_i in range(self.num_species):
                grid_i = self.species_grids[idx_i]
                params_i = self.species_params[idx_i]
                births = (np.random.rand(self.grid_size, self.grid_size) < params_i['birth_rate']) * grid_i
                deaths = (np.random.rand(self.grid_size, self.grid_size) < params_i['death_rate']) * grid_i
                grid_i += births.astype(int)
                grid_i -= deaths.astype(int)
                grid_i = np.clip(grid_i, 0, None)
                self.species_grids[idx_i] = grid_i

            for idx_i in range(self.num_species):
                for idx_j in range(self.num_species):
                    if idx_i == idx_j:
                        continue
                    effect = self.interaction_matrix[idx_i, idx_j]
                    if effect == 0:
                        continue
                    grid_i = self.species_grids[idx_i]
                    grid_j = self.species_grids[idx_j]
                    interactions = np.minimum(grid_i, grid_j)
                    if effect > 0:
                        benefit = (np.random.rand(self.grid_size, self.grid_size) < effect) * interactions
                        grid_i += benefit.astype(int)
                    elif effect < 0:
                        harm = (np.random.rand(self.grid_size, self.grid_size) < -effect) * interactions
                        grid_i -= harm.astype(int)
                        grid_i = np.clip(grid_i, 0, None)
                    self.species_grids[idx_i] = grid_i

            for idx in range(self.num_species):
                total_population = np.sum(self.species_grids[idx])
                self.population_counts[idx].append(total_population)

            img = np.zeros((self.grid_size, self.grid_size, 3))
            for idx, params in enumerate(self.species_params):
                grid = self.species_grids[idx]
                color = np.array(mcolors.to_rgb(params['color']))
                if grid.max() > 0:
                    normalized_grid = grid / grid.max()
                    img += np.outer(normalized_grid, color).reshape(self.grid_size, self.grid_size, 3)
            img = np.clip(img, 0, 1)
            im.set_data(img)
            ax1.set_title(f"Step {step+1}")

            for idx, line in enumerate(population_lines):
                line.set_data(range(len(self.population_counts[idx])), self.population_counts[idx])
            ax2.set_xlim(0, self.steps)
            max_population = max([max(counts) for counts in self.population_counts]) * 1.1
            ax2.set_ylim(0, max_population)

            clear_output(wait=True)
            display.display(fig)
            plt.pause(0.001)

        plt.close(fig)
