import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from IPython import display

class PredatorPreySimulation:
    """
    Simulates a predator-prey ecosystem on a 2D grid using simple rules for
    reproduction, death, and movement. Visualizes the state of the system
    and tracks population dynamics over time.
    """
    
    def __init__(self, grid_size=50, initial_prey=500, initial_predators=50,
                 prey_birth_rate=0.05, prey_death_rate=0.01,
                 predator_death_rate=0.05, predator_reproduction_rate=0.5,
                 steps=200):
        """
        Initialize the simulation with the given parameters for grid size, population counts, 
        and reproduction/death rates. Sets up the grid for prey and predator populations.
        
        :param grid_size: Size of the square grid.
        :param initial_prey: Initial number of prey.
        :param initial_predators: Initial number of predators.
        :param prey_birth_rate: Probability of prey reproducing in each step.
        :param prey_death_rate: Probability of prey dying naturally in each step.
        :param predator_death_rate: Probability of predators dying naturally in each step.
        :param predator_reproduction_rate: Probability of predators reproducing after eating prey.
        :param steps: Number of steps to simulate.
        """
        # Simulation parameters
        self.grid_size = grid_size
        self.initial_prey = initial_prey
        self.initial_predators = initial_predators
        self.prey_birth_rate = prey_birth_rate
        self.prey_death_rate = prey_death_rate
        self.predator_death_rate = predator_death_rate
        self.predator_reproduction_rate = predator_reproduction_rate
        self.steps = steps

        # Initialize grids and population counts
        self.prey_grid = np.zeros((grid_size, grid_size), dtype=int) 
        self.predator_grid = np.zeros((grid_size, grid_size), dtype=int) 
        self.prey_counts = [] 
        self.predator_counts = [] 

        # Populate the grid with initial prey and predators
        self.initialize_population()

    def initialize_population(self):
        """
        Randomly places the initial prey and predator populations on the grid.
        """
        # Randomly place initial prey
        for _ in range(self.initial_prey):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.prey_grid[x, y] += 1  

        # Randomly place initial predators
        for _ in range(self.initial_predators):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.predator_grid[x, y] += 1  

        # Record initial populations
        self.prey_counts.append(self.initial_prey)
        self.predator_counts.append(self.initial_predators)

    def move_agents(self, grid):
        """
        Moves prey or predators on the grid to random neighboring cells.

        :param grid: 2D array representing the population of prey or predators.
        :return: New grid after movement.
        """
        new_grid = np.zeros_like(grid)  
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                count = grid[x, y]  
                for _ in range(count):
                    # Move to a random neighboring cell (including staying in place)
                    dx, dy = np.random.choice([-1, 0, 1], size=2)
                    new_x = (x + dx) % self.grid_size 
                    new_y = (y + dy) % self.grid_size
                    new_grid[new_x, new_y] += 1  
        return new_grid

    def simulate(self):
        """
        Runs the predator-prey simulation for a specified number of steps. At each step, 
        prey reproduce, predators eat prey, and natural death occurs. Population dynamics 
        are visualized, and the grid is updated in real-time.
        """
        # Set up the figure and axes with adjusted width ratios
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})
        plt.subplots_adjust(wspace=0.4)  

        # Initialize grid map (for prey and predators visualization)
        img = np.zeros((self.grid_size, self.grid_size, 3))  
        im = ax1.imshow(img)
        ax1.set_title("Initial State")
        ax1.axis('off')  

        # Initialize population trends plot
        line_prey, = ax2.plot([], [], label='Prey Population', color='green')  
        line_predator, = ax2.plot([], [], label='Predator Population', color='red') 
        ax2.set_xlim(0, self.steps) 
        ax2.set_ylim(0, max(self.initial_prey, self.initial_predators) * 1.2)  
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Population')
        ax2.set_title('Population Trends Over Time')
        ax2.legend()

        # Main simulation loop
        for step in range(self.steps):
            # Move prey and predators
            self.prey_grid = self.move_agents(self.prey_grid)
            self.predator_grid = self.move_agents(self.predator_grid)

            # Prey reproduction
            prey_births = (np.random.rand(self.grid_size, self.grid_size) < self.prey_birth_rate) * self.prey_grid
            self.prey_grid += prey_births.astype(int)

            # Natural prey death
            prey_deaths = (np.random.rand(self.grid_size, self.grid_size) < self.prey_death_rate) * self.prey_grid
            self.prey_grid -= prey_deaths.astype(int)
            self.prey_grid = np.clip(self.prey_grid, 0, None)  

            # Predators eat prey
            encounters = np.minimum(self.prey_grid, self.predator_grid) 
            self.prey_grid -= encounters  # Prey eaten by predators

            # Predators reproduce based on encounters
            predator_births = (np.random.rand(self.grid_size, self.grid_size) < self.predator_reproduction_rate) * encounters
            self.predator_grid += predator_births.astype(int)

            # Natural predator death
            predator_deaths = (np.random.rand(self.grid_size, self.grid_size) < self.predator_death_rate) * self.predator_grid
            self.predator_grid -= predator_deaths.astype(int)
            self.predator_grid = np.clip(self.predator_grid, 0, None) 

            # Update population counts
            total_prey = np.sum(self.prey_grid) 
            total_predators = np.sum(self.predator_grid)  
            self.prey_counts.append(total_prey)
            self.predator_counts.append(total_predators)

            # Update grid map visualization
            img = np.zeros((self.grid_size, self.grid_size, 3)) 
            if self.predator_grid.max() > 0:
                img[:, :, 0] = self.predator_grid / self.predator_grid.max() 
            if self.prey_grid.max() > 0:
                img[:, :, 1] = self.prey_grid / self.prey_grid.max()
            im.set_data(img)
            ax1.set_title(f"Step {step + 1}")

            # Update population trends
            line_prey.set_data(range(len(self.prey_counts)), self.prey_counts) 
            line_predator.set_data(range(len(self.predator_counts)), self.predator_counts)  
            ax2.set_xlim(0, self.steps)
            max_population = max(max(self.prey_counts), max(self.predator_counts)) * 1.1  
            ax2.set_ylim(0, max_population)

            # Redraw the plots
            clear_output(wait=True)
            fig.canvas.draw()  
            display.display(fig)
            plt.pause(0.001)  

        plt.close(fig)  

        