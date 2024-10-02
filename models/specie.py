import numpy as np

class Specie:
    def __init__(self, config_specie):
        self.name = config_specie['name']
        self.speed = config_specie['speed']
        self.initial_energy = 100
        self.pack_size = config_specie['pack_size']
        self.color = config_specie['color']

        # Predators and Diet
        self.predators = config_specie['predators']
        self.diet = config_specie['diet']

        # Energy
        self.energy_decay = config_specie['energy_decay']
        self.movement_energy_cost = 0.1
        self.hunger_energy_penalty = config_specie.get('hunger_energy_penalty', 10.0)

        # Reproduction
        self.reproduction_energy_threshold = config_specie['reproduction_energy_threshold']
        self.reproduction_probability = config_specie.get('reproduction_probability', 1.0)

        self.mating_energy_decay = 0.9
        self.gestation_period = config_specie['gestation_period']
        self.birth_energy_cost = config_specie['birth_energy_cost']
        self.min_reproduction_age = config_specie['min_reproduction_age']
        self.max_reproduction_age = config_specie['max_reproduction_age']
        self.litter_size = config_specie['litter_size']
        self.mating_weight = config_specie['mating_weight']

        # Age
        self.max_age = config_specie['max_age']
        self.visual = 50  

        # Movement Weights (Default Values)
        self.separation_distance = 5.0  
        self.separation_weight = config_specie.get('separation_weight', 1.0)
        self.alignment_weight = config_specie.get('alignment_weight', 1.0)
        self.cohesion_weight = config_specie.get('cohesion_weight', 1.0)
        self.avoidance_weight = 1.0
        self.food_weight = config_specie.get('food_weight', 1.0)
        self.food_weight_multiplier = config_specie.get('food_weight_multiplier', 1.0) 
        self.random_movement_weight = 0.2
        self.max_avoidance_weight = 2.0
        self.max_food_weight = 2.0
        self.boundary_avoidance_weight = config_specie.get('boundary_avoidance_weight', 1.0)

        # Random death probability
        self.death_probability = config_specie.get('death_probability', 0.01)  
