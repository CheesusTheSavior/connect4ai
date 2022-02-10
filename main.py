import os
import pickle
import time
from itertools import combinations
from neuralnetwork import *
from connect4 import *
import numpy as np
import pandas as pd

with open("Names.txt", "r") as f:
    names = f.readlines()

names = [x[:-1] for x in names]


def random_name(names_list):
    name = np.random.choice(names_list)
    return name


def new_generation(names_list, size=10, bot_df: pd.DataFrame = None):
    bot_list = []
    first_gen = False

    for i in range(size):
        new_bot = GamBot()
        new_bot.name = random_name(names_list)
        bot_list.append(new_bot)

    if not bot_df:
        bot_df = pd.DataFrame()
        gen = 1
        first_gen = True
    else:
        gen = max(bot_df["gen"]+1)

    for bot in bot_list:
        # check if name is already in bot_df
        if first_gen:
            new_id = 0
            first_gen = False
        else:
            new_id = max(bot_df.index) + 1

        bot.name = f"{bot.name} #{new_id:05d}"
        bot_dict = dict(gen=gen, parent_1=None, parent_2=None,
                        bot=bot, status="alive", wins=0, loses=0, w_per_l=0, plays=0, score=0, max_score=0)
        bot_df = bot_df.append(bot_dict, ignore_index=True)

    return bot_df


def tournament(bot_df, hist_df=None):
    tour_df = bot_df[bot_df["status"] == "alive"]
    games = list(combinations(tour_df.index, 2))

    if not isinstance(hist_df, pd.DataFrame):
        hist_df = pd.DataFrame()
        round = 1
    else:
        round = max(hist_df["round"])+1

    for game in games:
        # get bots
        bot_1 = tour_df.at[game[0], "bot"]
        bot_2 = tour_df.at[game[1], "bot"]

        # generate a connect4 instance
        con4 = ConnectFour(f"{bot_1.name} vs {bot_2.name}")

        # let the bots play against each other
        print(f"{bot_1.name} is playing against {bot_2.name}!")
        playtime = time.ctime(time.time())
        con4.ai_vs_ai(bot_1, bot_2)

        # write down the winner or, if there is a tie, write down tie
        if con4.status == "finished":
            winner = con4.winner
            winner_name = bot_1.name if winner == 1 else bot_2.name
            print(f"{winner_name} has won!")
        elif con4.status == "tie":
            winner, winner_name = None, None
            print(f"Wow! The game ended in a tie!")
        else:
            print("Something must have gone wrong...")
            winner, winner_name = None, None

        # write important data about the game to dict
        game_dict = dict(p1=game[0], p2=game[1], p1_name=bot_1.name, p2_name=bot_2.name, time=playtime, winner=winner,
                         round=round)

        # write dict to hist_df
        hist_df = hist_df.append(game_dict, ignore_index=True)

        hist_df["p1"] = [int(x) for x in hist_df["p1"]]
        hist_df["p2"] = [int(x) for x in hist_df["p2"]]
        hist_df["winner"] = [int(x) for x in hist_df["winner"]]

    return hist_df


def save_bots(bot_df, filename, overwrite: bool = False):
    """
    Function for quickly pickling bot dataframes
    :param overwrite: bool to determine whether file shall be overwritten
    :param bot_df: DataFrame containing GamBots and data on them
    :param filename: string name for filename
    :return: does not return anything
    """
    if os.path.isfile(f"bot_library\\{filename}.txt") and overwrite == False:
        user = input("This file exists already. Continue? y/n")
        if user != "y":
            print("Aborting save.")
            return False

    with open(f"bot_library\\{filename}.txt", "wb") as my_file:
        pickle.dump(bot_df, my_file)
        print("bot_df saved to file.")

    return True


def load_bots(filename):
    if not os.path.isfile(f"bot_library\\{filename}.txt"):
        raise ValueError(f"File {filename}.txt not found!")

    else:
        with open(f"bot_library\\{filename}.txt", "rb") as my_file:
            bot_df = pickle.load(my_file)

    return bot_df


def calculate_scores(bot_df, hist_df):
    """
    Function to calculate scores from hist_df and update the values in bot_df
    :param bot_df:
    :param hist_df:
    :return:
    """
    current_round = hist_df[hist_df["round"]==max(hist_df["round"])]
    for row in current_round.index:
        p1 = hist_df.at[row, "p1"]
        p2 = hist_df.at[row, "p2"]
        bot_df.at[p1, "plays"] += 1
        bot_df.at[p2, "plays"] += 1

        game_winner = hist_df.at[row, "winner"]
        if game_winner is None:
            bot_df.at[p1, "score"] += 1
            bot_df.at[p2, "score"] += 1

            continue
        elif game_winner == 1:
            bot_df.at[p1, "score"] += 3
            bot_df.at[p1, "wins"] += 1
            bot_df.at[p2, "loses"] += 1
            continue
        elif game_winner == 2:
            bot_df.at[p2, "score"] += 3
            bot_df.at[p2, "wins"] += 1
            bot_df.at[p1, "loses"] += 1
            continue

        else:
            print("What the hell is even that.")

    for row in bot_df.index:
        if bot_df.at[row, "loses"] == 0:
            bot_df.at[row, "w_per_l"] = np.inf
        else:
            bot_df.at[row, "w_per_l"] = bot_df.at[row, "wins"] / bot_df.at[row, "loses"]

    bot_df = bot_df.sort_values(by="w_per_l", ascending=False)

    for row in bot_df.index:
        bot_df.at[row, "max_score"] = max((bot_df.at[row, "score"], bot_df.at[row, "max_score"]))

    return bot_df


def procreate(bot_df, procreation_quantity=10, names_list=None):
    if names_list is None:
        names_list = names
    bot_df = bot_df.sort_values(by="score", ascending=False)
    bot_df["status"] = "deceased"
    id_to_procreate = list(bot_df.head(procreation_quantity).index)
    for survivor in id_to_procreate:
        bot_df.loc[survivor, "status"] = "alive"

    to_procreate = list(bot_df.head(procreation_quantity)["bot"])
    # generate randomized coupling from procreation quantity
    # make sure that each parent gets two children
    np.random.shuffle(to_procreate)
    couples = [(to_procreate[i-1], to_procreate[i]) for i in range(len(to_procreate))]

    gen = max(bot_df["gen"])+1

    # loop through couples
    for couple in couples:

        # cross the parents
        child = couple[0].cross(couple[1])
        # mutate the child
        child.full_mutate()
        # generate new dictionary

        new_id = max(bot_df.index) + 1
        name = random_name(names_list)
        child.name = f"{name} #{new_id:05d}"
        child_dict = dict(gen=gen, parent_1=couple[0].name, parent_2=couple[1].name,
                        bot=child, status="alive", wins=0, loses=0, w_per_l=0, plays=0, score=0, max_score=0)

        bot_df = bot_df.append(child_dict, ignore_index=True)

    return bot_df

"""
Start procedure here
1. Generate new generation and save individuals in bot_df
2. Perform tournament (every living individual against each other) and save results in game_df
3. Update scoring in bot_df 
4. Kill worst 50% of living individuals
"""
