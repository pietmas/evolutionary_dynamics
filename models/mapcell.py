import json
import webcolors
import numpy as np

class MapCells:
    def __init__(self, map_jason, map_settings):
        self.map = map_jason
        self.settings = map_settings
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.dict_map = {}
        self.color_to_type = {mapType["color"]: mapType["name"] for mapType in self.settings}
        self.type_to_food = {
            "plain": {"grass": 100},
            "forest": {"berries": 100, "grass": 50},
            "savana": {"grass": 50, "bush": 50}
        }

        # Initialize the map with food quantities
        self.initialize_food_quantities()

    def initialize_food_quantities(self):
        for y in range(self.height):
            for x in range(self.width):
                cell_type = self.get_cell_type(x, y)
                food_types = self.type_to_food.get(cell_type, {})
                food_quantities = {food: quantity for food, quantity in food_types.items()}
                self.dict_map[(x, y)] = {
                    "color": self.map[y][x],
                    "type": cell_type,
                    "food": food_quantities,
                    "animals": []
                }

    def get_settings(self):
        return self.settings

    def get_map_size(self):
        return self.width, self.height

    def get_cell_color(self, x, y):
        try:
            return webcolors.hex_to_name(self.map[y][x])
        except:
            rgb_color = webcolors.hex_to_rgb(self.map[y][x])

            min_colors = {}
            for name in webcolors.CSS3_NAMES_TO_HEX:
                r_c, g_c, b_c = webcolors.hex_to_rgb(webcolors.CSS3_NAMES_TO_HEX[name])
                rd = (r_c - rgb_color[0]) ** 2
                gd = (g_c - rgb_color[1]) ** 2
                bd = (b_c - rgb_color[2]) ** 2
                min_colors[(rd + gd + bd)] = name

            return min_colors[min(min_colors.keys())]

    def get_cell_type(self, x, y):
        color = self.map[y][x]
        return self.color_to_type.get(color, "unknown")

    def get_food(self, x, y):
        return self.dict_map[(x, y)].get("food", {})

    def consume_food(self, x, y, food_name, quantity=1):
        if (x, y) in self.dict_map:
            if food_name in self.dict_map[(x, y)]["food"]:
                self.dict_map[(x, y)]["food"][food_name] -= quantity
                if self.dict_map[(x, y)]["food"][food_name] <= 0:
                    del self.dict_map[(x, y)]["food"][food_name]

    def regenerate_food(self):
        # Optionally implement food regeneration logic here
        pass

    def get_map(self):
        return self.dict_map
