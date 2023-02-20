import random, time
import numpy as np


# the tic tac toe board
class TTT_Board:
    def __init__(
        self, game_id, curr_user_id, X_id, O_id, move, turn_id, rows=3, cols=3,
    ):
        self.game_id = game_id
        self.curr_user_id = curr_user_id

        self.X_id, self.O_id = X_id, O_id
        self.opp_id = self.X_id if self.O_id == self.curr_user_id else self.O_id
        self.user_text = "X" if self.curr_user_id == self.X_id else "O"
        self.opp_text = "X" if self.user_text == "O" else "O"
        self.turn_id = turn_id
        self.turn = "X" if self.X_id == self.turn_id else "O"

        self.rows, self.cols = rows, cols

        self.board = []
        self.move_req = move

        self.scores = {self.user_text: 1, self.opp_text: -1, "tie": 0}

        self.generate_board()

        if self.turn_id == self.curr_user_id:
            self.move_req(self.game_id, random.randint(0, self.rows * self.cols - 1))

    def generate_board(self):
        for i in range(self.rows):
            self.board.append([])
            for j in range(self.cols):
                self.board[i].append(None)

    def move(self):
        # AI to make its turn
        best_score = -float("inf")
        best_move = None

        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] is None:
                    self.board[r][c] = self.user_text

                    # now itll be humans turn
                    score = self.minimax(0, False, self.opp_text, ["X", "O"],)

                    self.board[r][c] = None

                    if score > best_score:
                        best_score = score
                        best_move = (r, c)

        self.move_req(self.game_id, self.cols * best_move[0] + best_move[1])
        self.turn = self.opp_text
        self.turn_id = self.opp_id

    def place(self, data):
        id = data["to"]  # index for the 1d array
        # but our array is 2d, so convert it to 2d
        r, c = id // self.cols, id % self.cols
        text = data["turn_string"]

        # print(f"[{'BOT'}]: {text}({data['who']}) moved in TTT to {id}")
        self.board[r][c] = text

        self.turn_id = data["turn_id"]
        self.turn = "X" if self.turn_id == self.X_id else "O"
        if self.turn_id == self.curr_user_id:
            self.move()

    # check for game over (dynamic!)
    def check_game_over(self):
        # check for a complete row
        for r in range(self.rows):
            if len(set(self.board[r])) == 1 and self.board[r][0] is not None:
                return self.board[r][0]

        # check for a complete column
        for c in range(self.cols):
            col = set([self.board[r][c] for r in range(self.rows)])
            if len(col) == 1 and self.board[0][c] is not None:
                return self.board[0][c]

        # check for diagonals, but only if the game board is square
        if self.rows == self.cols:
            d1 = []  # diagonal from top left to botton right
            d2 = []  # diagonal from top right to bottom left
            for r in range(self.rows):
                d1.append(self.board[r][r])
                d2.append(self.board[r][self.rows - r - 1])

            if len(set(d1)) == 1 and d1[0] is not None:
                return d1[0]

            if len(set(d2)) == 1 and d2[0] is not None:
                return d2[0]

        # it is a tie
        none_count = 0
        for row in self.board:
            none_count += row.count(None)
        if none_count == 0:
            return "tie"

        return None  # game is not done yet

    def game_over_protocol(self, indices, winner_id, *args):
        print(f"[BOT]: GAME OVER {winner_id} won!")

    def minimax(
        self, depth, is_maximizing, turn, turns=["X", "O"],
    ):

        result = self.check_game_over()

        # terminal case, game is over so return the score corresponding to the player who won
        if result is not None:
            return self.scores[result]

        next_turn = turns[1] if turn == turns[0] else turns[0]

        # it is this players turn, we need to try and get the maximum score
        if is_maximizing:
            best_score = -float("inf")

            for r in range(self.rows):
                for c in range(self.cols):
                    if self.board[r][c] is None:
                        self.board[r][c] = turn
                        score = self.minimax(depth + 1, False, next_turn, turns,)
                        self.board[r][c] = None
                        best_score = max(score, best_score)

            return best_score

        # the human player will make a move with the least socre, i.e the best move possible for him
        else:
            best_score = float("inf")

            for r in range(self.rows):
                for c in range(self.cols):
                    if self.board[r][c] is None:
                        self.board[r][c] = turn
                        score = self.minimax(depth + 1, True, next_turn, turns)
                        self.board[r][c] = None
                        best_score = min(score, best_score)

            return best_score


class ConnectAI:
    def __init__(
        self, game_id, player_id, opp_id, turn_id, move_req, rows=6, cols=7,
    ):
        self.rows, self.cols = rows, cols  # dimensions of the board
        self.connect_n = 4  # how many coins need to be connected to win?

        # ids
        self.game_id = game_id
        self.player_id = int(player_id)
        self.opp_id = int(opp_id)
        self.turn_id = int(turn_id)
        self.players = [self.player_id, self.opp_id]

        self.move_req = move_req  # send a move request to the server

        # 2d game board
        # 0 represents an empty spot. spots filled with player ids are owned by those players
        self.board = np.array(
            [[0 for _ in range(self.cols)] for __ in range(self.rows)]
        )
        self.board_ = self.board.copy()

        # award payload
        self.score_system = {self.player_id: 1, self.opp_id: -1, "tie": 0, False: 0}

        # if it is this bots turn, then go ahead and play
        if self.turn_id == self.player_id:
            self.play(self.get_random_move())

        s = time.time()
        print(self.is_game_over())
        print(time.time() - s)

    # check if the current board state represents an end state
    def is_game_over(self):

        # check if some one has won
        for r in range(self.rows):
            for c in range(self.cols):

                if c <= self.cols - self.connect_n:
                    unique_coins_horizontal = np.unique(
                        self.board[r][c : c + self.connect_n]
                    )
                    # connect_n coins found horizontally
                    if (
                        len(unique_coins_horizontal) == 1
                        and unique_coins_horizontal[0] != 0
                    ):
                        return unique_coins_horizontal[0]

                if r <= self.rows - self.connect_n:
                    unique_coins_vertical = np.unique(
                        self.board[r : r + self.connect_n, c]
                    )
                    # connect_n coins found vertically
                    if (
                        len(unique_coins_vertical) == 1
                        and unique_coins_vertical[0] != 0
                    ):
                        return unique_coins_vertical[0]

                    # diagonals
                    if c <= self.cols - self.connect_n:
                        # connect_n x connect_n square. we care abt the 2 diagonals of this square
                        square = self.board[
                            r : r + self.connect_n, c : c + self.connect_n
                        ]

                        # top left -> bottom right diagonal
                        tlbr_unique = np.unique(square.diagonal())
                        if len(tlbr_unique) == 1 and tlbr_unique[0] != 0:
                            return tlbr_unique[0]

                        # mirror this square and find its diagonal
                        # top right -> bottom left
                        trbl_unique = np.unique(np.fliplr(square).diagonal())
                        if len(trbl_unique) == 1 and trbl_unique[0] != 0:
                            return trbl_unique[0]

        if 0 not in self.board:
            # no empty spot left *and* nobody has won
            return "tie"

        # game not over
        return False

    # evaluate a score for this game state
    def eval(self, winner=False):
        # if the game is over, then game_over passed may be one of [self.player_id,self.opp_id,"tie"]
        return self.score_system.get(winner)

    # get all possible moves for current game state
    def get_moves(self):

        for c in range(self.cols):
            # index 0 is the top row
            if self.board[0][c] == 0:
                yield c

    # get a random move
    def get_random_move(self):
        moves = [move for move in self.get_moves()]
        return random.choice(moves)

    # drop a coin in a column
    def _move(self, col, id):
        # loop thru the rows in reverse order. so the bottom one comes first
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = id
                return (r, col)

        raise Exception(f"FULL COLUMN {col}, id: {id}")

    def _undo_move(self, r, c):
        # loop thru the rows in reverse order. so the bottom one comes first
        if self.board[r][c] != 0:
            self.board[r][c] = 0
            return True

        raise Exception(f"EMPTY SPOT {r},{c}")

    def play(self, move=None):
        if move is None:
            alpha = -float("inf")  # worst possible score for maximizing player.
            beta = float("inf")  # worst possible score for minimizing player.
            max_depth = 10  # max depth to search to

            best_score, best_move, n = minimax(self, max_depth, alpha, beta, True)

            print(f"score is {best_score}, best move is {best_move}, n: {n}")

            # send the request to move to the server
            self.move_req(self.game_id, best_move)

        else:
            self.move_req(self.game_id, move)

    # place a coin in the desired location
    def place(self, data):
        row, col = data["to"]  # (row,col)
        mover_id = data["who"]

        # place the mover's id in that cell
        self.board[row][col] = mover_id

        self.turn_id = int(data["turn_id"])  # whose turn is it now

        # if it is this bots turn - play
        if self.turn_id == self.player_id:
            print("MY TURN")
            self.play()

    def game_over_protocol(self, indices, winner_id):
        print(f"[BOT]: {winner_id} has won!")


# minimax algorithm implementation, with alpha beta pruning
def minimax(game, depth, alpha, beta, isMaximizing):
    n = 1

    winner = game.is_game_over()
    if winner or depth == 0:
        # return score of the board state
        return game.eval(winner), None, n

    # the bots turn
    if isMaximizing:
        best_score = -float("inf")
        best_move = None

        for move in game.get_moves():
            # make this bot move
            row, col = game._move(move, game.player_id)

            # calculate the score from this branch
            # next turn will be of the minimizing player
            score, _, n_ = minimax(game, depth - 1, alpha, beta, False)
            n += n_

            game._undo_move(row, col)

            # if we have a better score than before...
            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            # beta cut off
            # if beta <= alpha:
            #     break

        return best_score, best_move, n

    # minimizing player
    else:
        best_score = float("inf")

        for move in game.get_moves():
            # make the opponents move
            row, col = game._move(move, game.opp_id)

            # calculate the score from this branch
            # next turn will be of the maximizing player
            score, _, n_ = minimax(game, depth - 1, alpha, beta, True)
            n += n_

            game._undo_move(row, col)

            # if we have a better score than before...
            best_score = min(best_score, score)
            beta = min(beta, score)

            # beta cut off
            # if beta <= alpha:
            #     break

        return best_score, None, n

