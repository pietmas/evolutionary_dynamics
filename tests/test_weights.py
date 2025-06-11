import os
import sys
import numpy as np
from types import SimpleNamespace

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.specie import Specie
from models.animal import Animal

class DummyEnv:
    def __init__(self):
        self.size = (100, 100)
        self.animal_instances = {1: SimpleNamespace(position=np.array([60, 60]))}

    def get_nearby_entities(self, animal):
        return {'animals': [], 'predators': {'pred': [1]}, 'food': {}}

    def move_animal(self, animal, new_position):
        pass

def test_weight_normalization_with_predators():
    spec_config = {
        'name': 'TestSpecies',
        'speed': 6,
        'energy_decay': 0.995,
        'max_age': 30,
        'min_reproduction_age': 5,
        'max_reproduction_age': 25,
        'reproduction_probability': 0.5,
        'reproduction_energy_threshold': 40,
        'gestation_period': 1,
        'birth_energy_cost': 20,
        'litter_size': 2,
        'mating_weight': 1.0,
        'predators': [],
        'diet': {},
        'pack_size': 10,
        'color': '#FF0000',
        'hunger_energy_penalty': 5.0,
        'death_probability': 0.02,
        'food_weight': 1.0,
        'food_weight_multiplier': 2.0,
    }

    specie = Specie(spec_config)
    env = DummyEnv()
    animal = Animal(specie, [50, 50], env)
    animal.energy = 50
    energy_before = animal.energy

    # Expected unnormalized weights
    avoidance = specie.max_avoidance_weight
    food_weight = max(specie.max_food_weight * (1 - energy_before / 100),
                      specie.food_weight)
    food_weight *= specie.food_weight_multiplier
    random_weight = specie.random_movement_weight

    weight_sum = (specie.separation_weight + specie.alignment_weight +
                  specie.cohesion_weight + avoidance + food_weight +
                  specie.mating_weight + specie.boundary_avoidance_weight +
                  random_weight)

    expected_weights = [
        specie.separation_weight / weight_sum,
        specie.alignment_weight / weight_sum,
        specie.cohesion_weight / weight_sum,
        avoidance / weight_sum,
        food_weight / weight_sum,
        specie.mating_weight / weight_sum,
        specie.boundary_avoidance_weight / weight_sum,
        random_weight / weight_sum,
    ]

    animal.move()

    actual_weights = [
        animal.separation_weight,
        animal.alignment_weight,
        animal.cohesion_weight,
        animal.avoidance_weight,
        animal.food_weight,
        animal.mating_weight,
        animal.border_avoidance_weight,
        specie.random_movement_weight / weight_sum,
    ]

    assert np.isclose(sum(actual_weights), 1.0)
    assert np.allclose(actual_weights, expected_weights)
