import numpy as np

class Animal:
    last_id = 0  # Class variable for assigning unique IDs

    def __init__(self, species, position, environment):
        self.specie = species
        self.energy = self.specie.initial_energy
        self.sex = np.random.choice(["F", "M"])
        self.age = 0
        self.position = np.array(position)
        self.can_reproduce = False
        self.environment = environment
        self.pregnancy_timer = 0
        self.is_pregnant = False
        self.velocity = np.array([0.0, 0.0])

        Animal.last_id += 1
        self.id = Animal.last_id

    def move(self):
        nearby_entities = self.environment.get_nearby_entities(self)
        nearby_animals_ids = nearby_entities['animals']
        nearby_predators = nearby_entities['predators']
        nearby_food = nearby_entities['food']

        nearby_animals = [self.environment.animal_instances[aid] for aid in nearby_animals_ids]

        # Adjust weights based on needs

        # Avoidance Weight: Increases when predators are nearby
        num_predators = sum(len(ids) for ids in nearby_predators.values())
        if num_predators > 0:
            self.avoidance_weight = self.specie.max_avoidance_weight
        else:
            self.avoidance_weight = 0.0

        # Increase food weight when energy is low
        self.food_weight = self.specie.max_food_weight * (1 - (self.energy / 100))
        self.food_weight = max(self.food_weight, self.specie.food_weight)

        # Adjust food weight based on species-specific multiplier
        self.food_weight *= self.specie.food_weight_multiplier  
    
        # Random Movement Weight
        random_movement_weight = self.specie.random_movement_weight
        

        weight_sum = (self.specie.separation_weight + self.specie.alignment_weight + self.specie.cohesion_weight +
                      self.specie.avoidance_weight + self.food_weight + self.specie.mating_weight +
                      self.specie.boundary_avoidance_weight + random_movement_weight)

        # Normalize weights
        self.separation_weight = self.specie.separation_weight / weight_sum
        self.alignment_weight = self.specie.alignment_weight / weight_sum
        self.cohesion_weight = self.specie.cohesion_weight / weight_sum
        self.avoidance_weight /= weight_sum
        self.food_weight /= weight_sum
        self.mating_weight = self.specie.mating_weight / weight_sum
        self.border_avoidance_weight = self.specie.boundary_avoidance_weight / weight_sum
        random_movement_weight /= weight_sum

        # Initialize movement vectors
        separation = np.array([0.0, 0.0])
        alignment = np.array([0.0, 0.0])
        cohesion = np.array([0.0, 0.0])
        avoidance = np.array([0.0, 0.0])
        food_attraction = np.array([0.0, 0.0])
        mating_attraction = np.array([0.0, 0.0])

        # Calculate Separation, Alignment, and Cohesion
        if nearby_animals:
            velocities = np.array([animal.velocity for animal in nearby_animals])
            positions = np.array([animal.position for animal in nearby_animals])

            delta_positions = self.position - positions
            distances_sq = np.sum(delta_positions**2, axis=1)
            separation_distance_sq = self.specie.separation_distance ** 2

            close_animals = (distances_sq < separation_distance_sq) & (distances_sq > 0)

            if np.any(close_animals):
                separation_vectors = delta_positions[close_animals] / np.sqrt(distances_sq[close_animals])[:, np.newaxis]
                separation = np.sum(separation_vectors, axis=0)
                if np.linalg.norm(separation) > 0:
                    separation = separation / np.linalg.norm(separation)

            avg_velocity = np.mean(velocities, axis=0)
            alignment = avg_velocity - self.velocity
            if np.linalg.norm(alignment) > 0:
                alignment = alignment / np.linalg.norm(alignment)

            avg_position = np.mean(positions, axis=0)
            cohesion = avg_position - self.position
            if np.linalg.norm(cohesion) > 0:
                cohesion = cohesion / np.linalg.norm(cohesion)

        # Calculate Avoidance from Predators
        if num_predators > 0:
            predator_positions = []
            for predator_ids in nearby_predators.values():
                predator_positions.extend([self.environment.animal_instances[pid].position for pid in predator_ids])
            predator_positions = np.array(predator_positions)
            delta_positions = self.position - predator_positions
            distances = np.sqrt(np.sum(delta_positions**2, axis=1))
            epsilon = 1e-6
            distances = np.where(distances == 0, epsilon, distances)

            avoidance_vectors = delta_positions / distances[:, np.newaxis]
            avoidance = np.sum(avoidance_vectors, axis=0)
            if np.linalg.norm(avoidance) > 0:
                avoidance = avoidance / np.linalg.norm(avoidance)

        # Calculate Food Attraction
        if nearby_food:
            preferred_food = max(
                nearby_food.keys(), key=lambda f: self.specie.diet[f]['energy']
            )
            food_items = nearby_food[preferred_food]
            food_positions = []

            for item in food_items:
                if isinstance(item, int):
                    food_animal = self.environment.animal_instances[item]
                    food_positions.append(food_animal.position)
                else:
                    food_positions.append(item)

            if food_positions:
                food_positions = np.array(food_positions)
                avg_food_pos = np.mean(food_positions, axis=0)
                food_attraction = avg_food_pos - self.position
                if np.linalg.norm(food_attraction) > 0:
                    food_attraction = food_attraction / np.linalg.norm(food_attraction)

        # Calculate Mating Attraction
        mating_attraction = np.array([0.0, 0.0])
        if self.can_reproduce:
            opposite_sex = "M" if self.sex == "F" else "F"
            nearby_animals_ids = nearby_entities['animals']
            nearby_animals = [self.environment.animal_instances[aid] for aid in nearby_animals_ids]
            potential_mates = [animal for animal in nearby_animals
                               if animal.sex == opposite_sex and animal.can_reproduce and animal.specie.name == self.specie.name]
            if potential_mates:
                mate_positions = np.array([mate.position for mate in potential_mates])
                avg_mate_pos = np.mean(mate_positions, axis=0)
                mating_attraction = avg_mate_pos - self.position
                if np.linalg.norm(mating_attraction) > 0:
                    mating_attraction = mating_attraction / np.linalg.norm(mating_attraction)

        # Random Movement
        random_movement = random_movement_weight * np.random.randn(2)
        if np.linalg.norm(random_movement) > 0:
            random_movement = random_movement / np.linalg.norm(random_movement)

        # Calculate Boundary Avoidance
        boundary_threshold = 5.0  # Adjust as needed
        boundary_avoidance = np.array([0.0, 0.0])

        # Left boundary
        distance_left = self.position[0]
        if distance_left < boundary_threshold:
            boundary_avoidance[0] += (boundary_threshold - distance_left) / boundary_threshold

        # Right boundary
        distance_right = self.environment.size[0] - self.position[0]
        if distance_right < boundary_threshold:
            boundary_avoidance[0] -= (boundary_threshold - distance_right) / boundary_threshold

        # Top boundary
        distance_top = self.position[1]
        if distance_top < boundary_threshold:
            boundary_avoidance[1] += (boundary_threshold - distance_top) / boundary_threshold

        # Bottom boundary
        distance_bottom = self.environment.size[1] - self.position[1]
        if distance_bottom < boundary_threshold:
            boundary_avoidance[1] -= (boundary_threshold - distance_bottom) / boundary_threshold

        if np.linalg.norm(boundary_avoidance) > 0:
            boundary_avoidance = boundary_avoidance / np.linalg.norm(boundary_avoidance)

        # Combine Movement Vectors with Weights
        movement = (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion +
            self.avoidance_weight * avoidance +
            self.food_weight * food_attraction +
            self.mating_weight * mating_attraction +
            self.border_avoidance_weight * boundary_avoidance +
            random_movement
        )

        # Normalize and Apply Speed
        norm = np.linalg.norm(movement)
        if norm > 0:
            movement = movement / norm

        movement *= self.specie.speed

        # Update Position
        new_position = np.round(self.position + movement)
        env_size = np.array(self.environment.size)
        new_position = np.clip(new_position, [0, 0], env_size - 1)

        self.environment.move_animal(self, new_position)
        self.velocity = movement
        self.energy -= self.specie.movement_energy_cost

    def check_reproduction(self):
        if not self.is_alive():
            self.can_reproduce = False
            return

        if self.age < self.specie.min_reproduction_age or self.age > self.specie.max_reproduction_age:
            self.can_reproduce = False
            return

        if self.energy < self.specie.reproduction_energy_threshold:
            self.can_reproduce = False
            return

        self.can_reproduce = True

    def handle_pregnancy(self):
        if self.is_pregnant:
            self.pregnancy_timer -= 1
            if self.pregnancy_timer <= 0:
                self.give_birth()

    def reproduce(self):
        if not self.is_alive() or not self.can_reproduce:
            return None

        # Reproduction occurs with probability self.specie.reproduction_probability
        if np.random.rand() > self.specie.reproduction_probability:
            return  # Reproduction does not occur

        opposite_sex = "M" if self.sex == "F" else "F"
        nearby_entities = self.environment.get_nearby_entities(self)
        nearby_animals_ids = nearby_entities['animals']
        nearby_animals = [self.environment.animal_instances[aid] for aid in nearby_animals_ids]
        potential_mates = [animal for animal in nearby_animals
                           if animal.sex == opposite_sex and animal.can_reproduce and animal.specie.name == self.specie.name]

        if potential_mates:
            mate = np.random.choice(potential_mates)
            self.energy *= self.specie.mating_energy_decay
            mate.energy *= mate.specie.mating_energy_decay

            if self.sex == "F":
                self.is_pregnant = True
                self.pregnancy_timer = self.specie.gestation_period
                self.can_reproduce = False
                mate.can_reproduce = False
            else:
                self.can_reproduce = False
                mate.is_pregnant = True
                mate.pregnancy_timer = mate.specie.gestation_period
                mate.can_reproduce = False

    def give_birth(self):
        num_offspring = np.random.randint(1, self.specie.litter_size + 1)
        for _ in range(num_offspring):
            child_position = self.position + np.random.uniform(-1, 1, 2)
            env_size = np.array(self.environment.size)
            child_position = np.clip(child_position, [0, 0], env_size - 1)

            child = Animal(self.specie, child_position, self.environment)
            self.environment.add_animal(child)
        self.energy -= self.specie.birth_energy_cost * num_offspring
        self.is_pregnant = False
        self.pregnancy_timer = 0

    def eat(self):
        self.ate = False  

        if self.is_alive():
            consumption_radius = 1.5  # Adjust as needed
            nearby_entities = self.environment.get_nearby_entities(self)
            nearby_food = nearby_entities['food']

            # Eat prey if available
            edible_prey_ids = []
            for prey_name, prey_ids in nearby_food.items():
                if prey_name in self.specie.diet and self.specie.diet[prey_name]['type'] == 'animal':
                    edible_prey_ids.extend(prey_ids)

            if edible_prey_ids:
                prey_positions = np.array([self.environment.animal_instances[pid].position for pid in edible_prey_ids])
                distances = np.linalg.norm(prey_positions - self.position, axis=1)
                close_prey_indices = np.where(distances <= consumption_radius)[0]

                if len(close_prey_indices) > 0:
                    prey_index = close_prey_indices[0]
                    prey_id = edible_prey_ids[prey_index]
                    prey_animal = self.environment.animal_instances[prey_id]
                    energy_gained = self.specie.diet[prey_animal.specie.name]["energy"]
                    self.energy = min(100, self.energy + energy_gained)
                    self.environment.remove_animal(prey_id)
                    self.ate = True  
            else:
                # Check for stationary food (e.g., plants)
                cell_contents = self.environment.get_food_at(self.position)
                edible_items = [item for item in cell_contents.keys() if item in self.specie.diet.keys()]
                for food in edible_items:
                    if cell_contents[food]["type"] == "vegetables":
                        if np.random.random() < self.specie.diet[food]["eating_probability"]:
                            energy_gained = self.specie.diet[food]["energy"]
                            self.energy = min(100, self.energy + energy_gained)
                            self.ate = True  
                            # Consume the food from the cell
                            self.environment.consume_food_at(self.position, food, quantity=1)
                            break

        if not self.ate:
            # Apply additional energy penalty if the animal did not eat
            self.energy -= self.specie.hunger_energy_penalty

    def update(self):
        self.move()
        self.eat()  

        self.energy *= self.specie.energy_decay
        self.age += 1

        if self.is_alive():
            self.check_reproduction()
            if self.sex == "F":
                self.handle_pregnancy()
            if self.can_reproduce and not self.is_pregnant:
                self.reproduce()

    def is_alive(self):
        # Random death rate implementation
        if np.random.rand() < self.specie.death_probability:
            return False
        return self.energy > 0 and self.age <= self.specie.max_age
