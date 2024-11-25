import numpy as np
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class Board:
    """
    Gomoku board class
    Board data:
    1=white, -1=black, 0=empty
    """

    def __init__(self, n=15):
        self.n = n
        # Create an empty board
        self.pieces = [[0] * n for _ in range(n)]

    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Return all legal move positions"""
        moves = []
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.append((x, y))
        return moves

    def has_legal_moves(self):
        """Check if there are any legal moves"""
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def execute_move(self, move, color):
        """Place a piece on the specified position"""
        x, y = move
        self[x][y] = color

    def is_win(self, color):
        """Check if there is a win"""
        # Check all directions
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    # Check each direction
                    for dx, dy in directions:
                        count = 1
                        # Check forward
                        tx, ty = x + dx, y + dy
                        while (
                            0 <= tx < self.n
                            and 0 <= ty < self.n
                            and self[tx][ty] == color
                        ):
                            count += 1
                            tx += dx
                            ty += dy
                        # Check backward
                        tx, ty = x - dx, y - dy
                        while (
                            0 <= tx < self.n
                            and 0 <= ty < self.n
                            and self[tx][ty] == color
                        ):
                            count += 1
                            tx -= dx
                            ty -= dy
                        if count >= 5:
                            return True
        return False


class GomokuGame:
    square_content = {-1: "X", +0: ".", +1: "O"}

    def __init__(self, n=15):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        valids = [0] * self.getActionSize()
        legalMoves = b.get_legal_moves(player)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if not b.has_legal_moves():
            return 0
        return None

    def getCanonicalForm(self, board, player):
        return player * board

    def getSymmetries(self, board, pi):
        # Gomoku's symmetries include rotation and reflection
        assert len(pi) == self.n**2
        pi_board = np.reshape(pi, (self.n, self.n))
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries += [(newB, newPi.ravel())]
        return symmetries

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("-")
        for y in range(n):
            print(f"{y:2d}|", end="")
            for x in range(n):
                piece = board[y][x]
                print(GomokuGame.square_content[piece], end=" ")
            print("|")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("-")
class RandomGomokuPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class GreedyGomokuPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]


class HumanGomokuPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / self.game.n), int(i % self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    if (
                        (0 <= x)
                        and (x < self.game.n)
                        and (0 <= y)
                        and (y < self.game.n)
                    ) or ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y
                        if valid[a]:
                            break
                except ValueError:
                    "Invalid integer"
            print("Invalid move")
        return a


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it. Is necessary for verbose
                     mode.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2, 0 if draw)
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1  # player1 go first
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) is None:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](
                self.game.getCanonicalForm(board, curPlayer)
            )

            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1
            )

            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(f"valids = {valids}")
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        result = curPlayer * self.game.getGameEnded(board, curPlayer)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(result))
            self.display(board)
        return result

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (player1 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (player2 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
