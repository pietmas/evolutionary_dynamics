import numpy as np
import json
from .animal import Animal
from .specie import Specie
import yaml
from PIL import Image, ImageDraw
import colorsys
import tkinter as tk
from PIL import ImageTk
from scipy.spatial import KDTree
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from IPython import display

class Environment:
    def __init__(self, map_cell):
        self.map_cell = map_cell
        self.size = map_cell.get_map_size()  
        self.map = map_cell.get_map()        
        self.width, self.height = self.size
        self.animals = {}  # id: position as numpy array
        self.animal_instances = {}  # id: Animal instance
        self.config = None
        self.tk_image = None

        # Spatial index
        self.kd_tree = None
        self.animal_positions = None
        self.animal_ids = None

        # Cache
        self.nearby_entities_cache = {}

    def initialize_animals(self, config):
        species_dict = {}
        self.config = config

        # Initialize each species from config
        for config_specie in self.config["species"]:
            species_name = config_specie["name"]
            species_dict[species_name] = Specie(config_specie)

        # Loop over species and create animals
        for specie_name, specie in species_dict.items():
            specie_count = self.config["simulation"]["initial_population"][specie_name]
            pack_size = specie.pack_size
            num_packs = max(1, specie_count // pack_size)
            remaining_animals = specie_count

            # Create packs of animals
            for _ in range(num_packs):
                pack_center = np.random.randint(low=0, high=[self.width, self.height])

                current_pack_size = min(pack_size, remaining_animals)

                for _ in range(current_pack_size):
                    offset = np.random.randint(low=-5, high=5, size=2)
                    position = pack_center + offset
                    position = np.clip(position, [0, 0], [self.width - 1, self.height - 1])

                    animal = Animal(specie, position, self)
                    self.add_animal(animal)
                    remaining_animals -= 1

                if remaining_animals <= 0:
                    break

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_food_at(self, position):
        x, y = int(position[0]), int(position[1])
        food = {}
        if self.is_within_bounds(x, y):
            map_food = self.map[(x, y)].get("food", {})
            for food_name, quantity in map_food.items():
                food[food_name] = {'quantity': quantity, 'type': 'vegetables'}
            animals = self.map[(x, y)].get("animals", [])
            for animal in animals:
                animal_name = animal.specie.name
                if animal_name != self.animal_instances[animal.id].specie.name:  # Exclude self
                    if animal_name not in food:
                        food[animal_name] = {'quantity': 0, 'type': 'animal', 'ids': []}
                    food[animal_name]['quantity'] += 1
                    food[animal_name]['ids'].append(animal.id)
        return food

    def consume_food_at(self, position, food_name, quantity=1):
        x, y = int(position[0]), int(position[1])
        self.map_cell.consume_food(x, y, food_name, quantity)

    def update_spatial_index(self):
        positions = []
        animal_ids = []
        for animal in self.animal_instances.values():
            positions.append(animal.position)
            animal_ids.append(animal.id)
        positions = np.array(positions)
        if len(positions) > 0:
            self.kd_tree = KDTree(positions)
        else:
            self.kd_tree = None
        self.animal_positions = positions
        self.animal_ids = np.array(animal_ids)
        self.nearby_entities_cache = {}  

    def get_nearby_entities(self, animal):
        if animal.id in self.nearby_entities_cache:
            return self.nearby_entities_cache[animal.id]

        nearby_entities = {
            'animals': [],
            'predators': {},
            'food': {}
        }

        if self.kd_tree is not None:
            radius = animal.specie.visual
            idxs = self.kd_tree.query_ball_point(animal.position, radius)
            idxs = [i for i in idxs if self.animal_ids[i] != animal.id]
            nearby_animals = []
            for i in idxs:
                animal_id = int(self.animal_ids[i])
                if animal_id in self.animal_instances:
                    other_animal = self.animal_instances[animal_id]
                    nearby_animals.append(other_animal)
                else:
                    pass  

            for other_animal in nearby_animals:
                if other_animal.specie.name == animal.specie.name:
                    nearby_entities['animals'].append(other_animal.id)
                else:
                    if animal.specie.name in other_animal.specie.diet:
                        nearby_entities['predators'].setdefault(other_animal.specie.name, []).append(other_animal.id)
                    if other_animal.specie.name in animal.specie.diet:
                        nearby_entities['food'].setdefault(other_animal.specie.name, []).append(other_animal.id)

        # Store in cache
        self.nearby_entities_cache[animal.id] = nearby_entities
        return nearby_entities

    def move_animal(self, animal, new_position):
        old_x, old_y = int(animal.position[0]), int(animal.position[1])
        new_x, new_y = int(new_position[0]), int(new_position[1])

        # Remove animal from old cell
        if self.is_within_bounds(old_x, old_y):
            if "animals" in self.map[(old_x, old_y)]:
                if animal in self.map[(old_x, old_y)]["animals"]:
                    self.map[(old_x, old_y)]["animals"].remove(animal)

        # Update animal's position
        animal.position = new_position
        self.animals[animal.id] = new_position

        # Add animal to new cell
        if self.is_within_bounds(new_x, new_y):
            if "animals" not in self.map[(new_x, new_y)]:
                self.map[(new_x, new_y)]["animals"] = []
            self.map[(new_x, new_y)]["animals"].append(animal)

    def add_animal(self, animal):
        x, y = int(animal.position[0]), int(animal.position[1])
        if self.is_within_bounds(x, y):
            if "animals" not in self.map[(x, y)]:
                self.map[(x, y)]["animals"] = []
            self.map[(x, y)]["animals"].append(animal)
            self.animals[animal.id] = animal.position
            self.animal_instances[animal.id] = animal

    def remove_animal(self, animal_id):
        animal = self.animal_instances.get(animal_id)
        if animal is None:
            return
        position = animal.position
        x, y = int(position[0]), int(position[1])
        if self.is_within_bounds(x, y):
            if self.map[(x, y)]["animals"]:
                if animal in self.map[(x, y)]["animals"]:
                    self.map[(x, y)]["animals"].remove(animal)
        del self.animals[animal_id]
        del self.animal_instances[animal_id]

    def generate_map_image(self):
        width, height = self.width, self.height
        map_array = np.zeros((height, width, 3), dtype=np.uint8)

        for (x, y), cell in self.map.items():
            color_code = cell["color"]
            color_rgb = tuple(int(color_code.strip('#')[i:i+2], 16) for i in (0, 2, 4))
            map_array[y, x] = color_rgb

        # Draw animals
        for animal in self.animal_instances.values():
            animal_x = int(round(animal.position[0]))
            animal_y = int(round(animal.position[1]))
            if 0 <= animal_x < width and 0 <= animal_y < height:
                animal_color_rgb = tuple(int(animal.specie.color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
                map_array[animal_y, animal_x] = animal_color_rgb

        # Scale up the image
        scale_factor = 2
        map_array = np.repeat(np.repeat(map_array, scale_factor, axis=0), scale_factor, axis=1)
        return map_array

    def count_species(self):
        species_counts = {}
        for animal in self.animal_instances.values():
            species = animal.specie.name
            species_counts[species] = species_counts.get(species, 0) + 1
        return species_counts

    def simulate(self):
        ages = self.config["simulation"]["ages"]
        population_trends = {}
        species_names = [specie['name'] for specie in self.config["species"]]

        # Create a mapping from species names to their colors
        species_colors = {specie['name']: specie['color'] for specie in self.config["species"]}

        # Initialize population trends
        for species in species_names:
            population_trends[species] = []
        x_data = []

        # Initialize matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Prepare population trend plot
        lines = {}
        for species in species_names:
            color = species_colors.get(species, '#000000')  
            lines[species], = ax2.plot([], [], label=species, color=color)
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Population')
        ax2.legend()
        ax2.set_title('Population Trend')

        for age in range(ages):
            print(f"Age: {age}")
            x_data.append(age)
            dead_animals = []

            # First, identify and remove dead animals
            for animal_id in list(self.animal_instances.keys()):
                animal = self.animal_instances[animal_id]
                if not animal.is_alive():
                    dead_animals.append(animal_id)
            for animal_id in dead_animals:
                self.remove_animal(animal_id)
            dead_animals.clear()

            # Now update the spatial index
            self.update_spatial_index()

            # Now update each animal
            for animal_id in list(self.animals.keys()):
                animal = self.animal_instances.get(animal_id)
                if animal and animal.is_alive():
                    animal.update()
                    self.animals[animal_id] = animal.position
                else:
                    dead_animals.append(animal_id)

            # Remove any animals that died during the update
            for animal_id in dead_animals:
                self.remove_animal(animal_id)
            dead_animals.clear()

            # Update population trends
            species_counts = self.count_species()
            for species in species_names:
                count = species_counts.get(species, 0)
                population_trends[species].append(count)
                # Update the line data
                lines[species].set_data(x_data, population_trends[species])

            # Update plot limits
            ax2.relim()
            ax2.autoscale_view()

            # Update map image
            map_image = self.generate_map_image()

            # Clear previous output and display the updated figures
            display.clear_output(wait=True)

            # Update map display
            ax1.clear()
            ax1.imshow(map_image)
            ax1.axis('off')
            ax1.set_title('Map')

            # Redraw the population trend
            ax2.relim()
            ax2.autoscale_view()
            for species in species_names:
                lines[species].set_data(x_data, population_trends[species])

            fig.canvas.draw()
            display.display(fig)

        plt.close(fig)
        
    def simulate_tkinter(self):
        ages = self.config["simulation"]["ages"]
        population_trends = {}

        root = tk.Tk()
        root.title("Environment Simulation")

        map_image = self.generate_map_image()
        self.tk_image = ImageTk.PhotoImage(map_image)
        label = tk.Label(root, image=self.tk_image)
        label.pack()

        for age in range(ages):
            print(f"Age: {age}")
            dead_animals = []

            # First, identify and remove dead animals
            for animal_id in list(self.animal_instances.keys()):
                animal = self.animal_instances[animal_id]
                if not animal.is_alive():
                    dead_animals.append(animal_id)
            for animal_id in dead_animals:
                self.remove_animal(animal_id)
            dead_animals.clear()

            # Now update the spatial index
            self.update_spatial_index()

            # Now update each animal
            for animal_id in list(self.animals.keys()):
                animal = self.animal_instances.get(animal_id)
                if animal and animal.is_alive():
                    animal.update()
                    self.animals[animal_id] = animal.position
                else:
                    dead_animals.append(animal_id)

            # Remove any animals that died during the update
            for animal_id in dead_animals:
                self.remove_animal(animal_id)
            dead_animals.clear()

            # Update visualization
            self.map_image = self.generate_map_image()
            self.tk_image = ImageTk.PhotoImage(self.map_image)
            label.configure(image=self.tk_image)
            label.image = self.tk_image
            root.update()

            species_counts = self.count_species()
            for species, count in species_counts.items():
                population_trends.setdefault(species, []).append(count)

        for species, counts in population_trends.items():
            print(f"Population trend for {species}: {counts}")

        root.destroy()