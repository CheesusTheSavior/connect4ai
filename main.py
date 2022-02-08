import pickle
from neuralnetwork import *
from connect4 import *
import numpy as np

with open("Names.txt", "r") as f:
    names = f.readlines()

names = [x[:-1] for x in names]


def random_name(names):
    name = np.random.choice(names)
    return name


def new_generation(size=10):
    bot_list = []

    for i in range(size):
        new_bot = GamBot()
        new_bot.name = random_name(names)
        bot_list.append(new_bot)

    return bot_list

"""
Start procedure here
1. Generate new generation and save individuals in bot_df
2. Perform tournament (every living individual against each other) and save results in game_df
3. Update scoring in bot_df 
4. Kill worst 50% of living individuals
"""