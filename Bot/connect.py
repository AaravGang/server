import pygame
import numpy as np

width, height = 800, 900
win = pygame.display.set_mode((width, height))

RED = (255, 0, 0)
YELLOW = (255, 255, 0)


class Button:
    def __init__(self, x, y, width, height, onclick, id=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (0, 0, 0)
        self.onclick = onclick
        self.id = id

        self.hover = False
        self.clicked = False

    def update(self):
        self.clicked = False
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            self.hover = True

            if pygame.mouse.get_pressed()[0]:
                self.onclick(self)
                self.clicked = True
        else:
            self.hover = False

    def show(self, win):
        pygame.draw.rect(win, self.color, self.rect)
        if self.hover:
            pygame.draw.rect(win, (255, 89, 234), self.rect)
        if self.clicked:
            pygame.draw.rect(win, (98, 98, 245), self.rect)
        pygame.draw.rect(win, (255, 255, 255), self.rect, 5)


class ConnectN:
    def __init__(self, rows=6, cols=7, n=4, width=width, height=height, x=0, y=0):
        self.rows, self.cols, self.n = rows, cols, n
        self.board = np.zeros((rows, cols), dtype=np.int64)

        self.width, self.height = width, height
        self.x, self.y = x, y
        self.surf = pygame.Surface((self.width, self.height))

        self.player = 1
        self.colors = [RED, YELLOW]

        self.switch_players_btn = Button(
            10, 10, self.width - 20, 80, self.switch_players
        )

        btn_startx, btn_starty = 0, 100
        btn_height = (self.height - btn_starty) // self.rows
        btn_width = (self.width - btn_startx) // self.cols

        self.buttons = [
            [
                Button(
                    btn_startx + btn_width * c,
                    btn_starty + btn_height * r,
                    btn_width,
                    btn_height,
                    self.place,
                    id=(r, c),
                )
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

        self.postion, self.mask = self.get_position_mask_bitmap()
        self.cache = {}

    def place(self, btn):
        self.board[btn.id[0]][btn.id[1]] = self.player
        btn.color = self.colors[self.player - 1]

    def switch_players(self, *args):
        self.player = 1 if self.player == 2 else 2

    def get_position_mask_bitmap(self,):
        position, mask = "", ""
        # Start with right-most column
        for j in range(self.cols - 1, -1, -1):
            # Add 0-bits to sentinel
            mask += "0"
            position += "0"
            # Start with bottom row
            for i in range(0, 6):
                mask += ["0", "1"][int(self.board[i, j] != 0)]
                position += ["0", "1"][int(self.board[i, j] == self.player)]

        return int(position, 2), int(mask, 2)

    # @staticmethod
    def is_game_over(self, position):
        # Horizontal check
        m = position & (position >> 7)
        if m & (m >> 14):
            return True

        # Diagonal \
        m = position & (position >> 6)

        s = "0" * (49 - len(bin(position)[2:])) + bin(position)[2:]
        b = np.fliplr(
            np.array(
                [[int(s[j * 7 + i], 2) for i in range(7)] for j in range(7)],
                dtype=np.int64,
            )
        )
        print(b)
        print("-" * 100)
        print(b >> 6)
        print("-" * 100)
        print(b >> 12)
        print("-" * 100)
        print((b >> 6) & (b >> 12))

        if m & (m >> 12):
            return True

        # Diagonal /
        m = position & (position >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = position & (position >> 1)
        if m & (m >> 2):
            return True
        # Nothing found
        return False

    @staticmethod
    def make_move(position, mask, col):
        new_position = position ^ mask
        new_mask = mask | (mask + (1 << (col * 7)))

        return new_position, new_mask

    def draw(self, win):
        self.surf.fill((128, 128, 128))
        self.switch_players_btn.update()
        self.switch_players_btn.show(self.surf)

        for row in self.buttons:
            for btn in row:
                btn.update()
                btn.show(self.surf)

        win.blit(self.surf, (self.x, self.y))


# minimax algorithm implementation, with alpha beta pruning
def minimax(game: ConnectN, depth, alpha, beta, isMaximizing, position, mask):
    if depth == 0:
        return 0

    game_over = game.is_game_over(position)
    if game_over:
        return int(not isMaximizing)

    # the bots turn
    if isMaximizing:
        best_score = -float("inf")
        best_move = None

        for move in game.get_moves():
            # make this bot move
            row, col = game._move(move, game.player_id)

            # calculate the score from this branch
            # next turn will be of the minimizing player
            score, _ = minimax(game, depth - 1, alpha, beta, False)

            game._undo_move(row, col)

            # if we have a better score than before...
            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            # pruning
            if beta <= alpha:
                break

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

            # pruning
            if beta <= alpha:
                break

        return best_score, None, n


def draw(board):
    win.fill((0, 0, 0))
    board.draw(win)
    pygame.display.update()


def main():
    board = ConnectN()

    run = True
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                break

            if e.type == pygame.KEYDOWN:
                print(board.is_game_over(board.get_position_mask_bitmap()[0]))

        draw(board)


main()
