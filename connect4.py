from itertools import cycle

import numpy as np
import pandas as pd


class ConnectFour:

    def __init__(self, name=None, rows=6, cols=7):
        if name:
            self.name = name

        self.game_rows = range(1, rows + 1)
        self.game_cols = range(1, cols + 1)
        self.game = pd.DataFrame(data="",
                                 index=self.game_rows,
                                 columns=self.game_cols)

        self.turns = cycle([1, 2])
        self.turn = next(self.turns)
        self.symbols = ["x", "o"]
        self.status = "start"
        self.end = False

    def show(self):
        with pd.option_context('display.max_rows',
                               None, 'display.max_columns', None):
            print(self.game)

    def take_turn(self):
        first_input = True
        valid = False
        cur_player = self.player_names[self.turn - 1]
        while valid == False:
            # take input from player (which column)
            if first_input:
                plays = input(f"{cur_player}, it is your turn!\n Choose your column:")
                first_input = False
            else:
                plays = input(f"{cur_player}, this move is not valid. Choose a different column:")
            if plays == "stop":
                return plays
            try:
                plays = int(plays)
            except ValueError:
                print("This is not a valid input. Input an integer.")
                continue
            # check if player's input is correct
            valid = True
            # 1. must be inside range of columns
            if plays not in self.game_cols:
                valid = False
                print("This column is not in range.")
                continue
            # 2. columns must not be full
            if not "" in list(self.game[plays]):
                valid = False
                print("This column is full.")
                continue

        # depending on player, drop a character in the lowest empty row
        play_col = list(self.game[plays])

        done = False
        i = -1
        symbol = self.symbols[self.turn - 1]
        while not done:
            if play_col[i] == "":
                play_col[i] = symbol
                self.game[plays] = play_col
                print(f"\nPlaced a {symbol} in column {plays}.\n")
                done = True

                return "play"
            else:
                i -= 1

    def check_winner(self):
        # check if there are four identical pieces in any
        # row, column or diagonal right next to each other
        # check whether the game has ended in a tie
        full = np.all(self.game.values != "")
        if full:
            self.status = "tie"
            return True

        # iterate through all cells

        for col in self.game_cols:
            for row in self.game_rows:
                cell_val = self.game.loc[row, col]
                if cell_val == "":
                    continue
                right_cols = list(range(col, col + 4))
                lower_rows = list(range(row, row + 4))
                right_cells, lower_cells, diagonal_cells, diagonal_cells2 = ([None], [None], [None], [None])
                if col + 3 in self.game_cols:
                    right_cells = list(self.game.loc[row, right_cols])
                if row + 3 in self.game_rows:
                    lower_cells = list(self.game.loc[lower_rows, col])
                if col + 3 in self.game_cols and row + 3 in self.game_rows:
                    diagonal_cells = []
                    for i in range(4):
                        diagonal_cells.append(self.game.at[row + i, col + i])
                if col + 3 in self.game_cols and row - 3 in self.game_rows:
                    diagonal_cells2 = []
                    for i in range(4):
                        diagonal_cells2.append(self.game.at[row - i, col + i])

                # check if the same symbol is in the next (right, lower or diagonal) three cells
                right_4, lower_4, diag_4, diag2_4 = (None, None, None, None)
                if right_cells:
                    right_4 = [True if x == cell_val else False for x in right_cells]
                if lower_cells:
                    lower_4 = [True if x == cell_val else False for x in lower_cells]
                if diagonal_cells:
                    diag_4 = [True if x == cell_val else False for x in diagonal_cells]
                if diagonal_cells2:
                    diag2_4 = [True if x == cell_val else False for x in diagonal_cells2]

                if all(right_4) or all(lower_4) or all(diag_4) or all(diag2_4):
                    self.winner = self.symbols.index(cell_val) + 1
                    self.status = "finished"
                    return True

        # if that is true, set winner and return true
        return False

    def play(self):
        command = None
        print("We are now playing CONNECT 4!!!")
        self.show()
        if not hasattr(self, "player_names"):
            pl1_name = input("Player 1, what is your name?\n")
            pl2_name = input("Player 2, what is your name?\n")
            self.player_names = [pl1_name, pl2_name]
        while self.check_winner() == False and command != "stop":
            command = self.take_turn()
            self.convert()
            self.show()

        if hasattr(self, "winner"):
            print(f"The winner is Player {self.winner}!")
        elif self.status == "tie":
            print(f"The game has ended in a tie!")
        else:
            print(f"No winner yet.")

    def reset(self):
        self.game = pd.DataFrame(data="",
                                 index=self.game_rows,
                                 columns=self.game_cols)
        if hasattr(self, "winner"):
            del (self.winner)
        del (self.player_names)
        self.turn = next(self.turns)
        self.status = "start"

    def convert(self):
        game = self.game.values
        shape = game.shape
        converted_game = np.zeros(shape)
        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                if game[x, y] == "":
                    converted_game[x, y] = 0
                elif game[x, y] == "x":
                    converted_game[x, y] = 1
                elif game[x, y] == "o":
                    converted_game[x, y] = -1
        self.conv_game = converted_game.reshape(42, 1)

    def play_vs_ai(self, bot):
        command = None
        print("We are now playing CONNECT 4!!!")
        self.show()
        if not hasattr(self, "player_names"):
            pl1_name = input("Player 1, what is your name?\n")
            pl2_name = bot.name
            self.player_names = [pl1_name, pl2_name]
        while self.check_winner() == False and command != "stop":
            if self.turn == 1:
                command = self.take_turn()
            else:
                self.bot_turn(bot)

            self.convert()
            self.show()
            self.turn = next(self.turns)

        if hasattr(self, "winner"):
            print(f"The winner is {self.winner}!")
        elif self.status == "tie":
            print(f"The game has ended in a tie!")
        else:
            print(f"No winner yet.")

    def bot_turn(self, bot):
        game_inputs = self.conv_game
        if self.turn == 2:
            game_inputs = [-x for x in game_inputs]
        # let the bot calculate his turn
        bot_activation = list(bot.calculate(game_inputs))
        # play the column with the highest amplitude
        print(bot_activation)
        bot_plays = [int(bot_activation.index(x)) + 1 for x in sorted(bot_activation, reverse=True)]
        # if that column is full, play the next highest amplitude
        i = 0
        valid = False
        while valid == False:
            # take input from player (which column)
            plays = bot_plays[i]
            print(f"{bot.name} plays {plays}")
            i += 1
            # check if player's input is correct
            valid = True
            # 2. columns must not be full
            if not "" in list(self.game[plays]):
                valid = False
                print("This column is full.")
                continue

            # depending on player, drop a character in the lowest empty row
            play_col = list(self.game[plays])

            done = False
            i = -1
            symbol = self.symbols[self.turn - 1]
            while not done:
                if play_col[i] == "":
                    play_col[i] = symbol
                    self.game[plays] = play_col
                    print(f"\nPlaced a {symbol} in column {plays}.\n")
                    done = True
                    return "play"
                else:
                    i -= 1
