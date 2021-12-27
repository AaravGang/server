# the tic tac toe board
class TTT_Board:
    def __init__(
        self, X_id, O_id, move, rows=3, cols=3,
    ):
        self.X_id, self.O_id = X_id, O_id
        self.rows, self.cols = rows, cols

        self.board = []
        self.move_req = move

        self.generate_board()

    def generate_board(self):
        for ind in range(self.rows * self.cols):
            self.board.append(0)

    def move(self):
        # flattened index
        self.move_req(1)

    def place(self, data):
        id = data["to"]
        text = data["turn_string"]
        # print(f"[{'BOT'}]: {text}({data['who']}) moved in TTT to {id}")
        self.board[id] = text

        self.move()

    def game_over_protocol(self, indices, winner_id, *args):
        print(f"[BOT]: GAME OVER {winner_id} won!")


# the connect4 board
class Connect4_Board:
    def __init__(
        self, curr_player_id, red_id, blue_id, move, rows=12, cols=13,
    ):

        self.red_id, self.blue_id = red_id, blue_id
        self.player_color = "red" if self.red_id == curr_player_id else "blue"

        self.rows, self.cols = rows, cols

        # the board - None is open, red is filled with red coin, and blue is filled with blue coin
        self.board = [[None for c in range(self.cols)] for r in range(self.rows)]

        self.move_req = move

        # variable to control the animating coin, when some one wins
        self.game_over = False
        self.winning_indices = []

    def move(self):
        # column to place
        self.move_req(1)

    # what happens when the game ends?
    def game_over_protocol(self, indices, winner_id, *args):
        print(f"[BOT]: GAME OVER {winner_id} won!")

    # place a coin in the desired location
    def place(self, data):
        to = data["to"]  # (row,col)
        text = data["turn_string"]
        self.board[to[0]][to[1]] = text

        # print(f"[{'BOT'}]: {text}({data['who']}) moved in CONNECT4 to {to}")
        self.move()

