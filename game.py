import numpy as np
import logging
from tqdm import tqdm
import pygame
import sys

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
        # Fix: Ensure consistent coordinate transformation
        move = (action // self.n, action % self.n)  # This gives (x, y)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        valids = [0] * self.getActionSize()
        legalMoves = b.get_legal_moves(player)
        for x, y in legalMoves:
            # Fix: Ensure consistent coordinate transformation
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
    def display(board, player1_first=True):
        """Update the GUI display"""
        if not hasattr(GomokuGame, 'gui'):
            GomokuGame.gui = GomokuGUI(len(board), player1_first)
        GomokuGame.gui.draw_board(board, player1_first)
        pygame.display.flip()


class GomokuGUI:
    def __init__(self, board_size, player1_first=True):
        pygame.init()
        self.board_size = board_size
        self.cell_size = 40
        self.margin = 40
        
        # Add space at the bottom for buttons and result text
        self.bottom_margin = 80  # Space for buttons at bottom
        
        # Calculate window size with bottom margin
        self.window_size = 2 * self.margin + self.cell_size * (self.board_size - 1)
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + self.bottom_margin))
        pygame.display.set_caption("AlphaZero Gomoku")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BROWN = (205, 170, 125)
        self.RED = (255, 0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        
        # Add button properties
        self.button_height = 30
        self.button_width = 100
        self.button_margin = 10
        
        # Position buttons at the bottom
        self.next_button = pygame.Rect(
            self.window_size - self.button_width - self.button_margin,
            self.window_size + (self.bottom_margin - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        self.quit_button = pygame.Rect(
            self.window_size - 2 * self.button_width - 2 * self.button_margin,
            self.window_size + (self.bottom_margin - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        self.player1_first = player1_first

    def get_mouse_position(self):
        """Convert mouse position to board coordinates"""
        x, y = pygame.mouse.get_pos()
        board_x = int((x - self.margin + self.cell_size/2) / self.cell_size)
        board_y = int((y - self.margin + self.cell_size/2) / self.cell_size)
        if 0 <= board_x < self.board_size and 0 <= board_y < self.board_size:
            return board_x, board_y
        return None

    def draw_board(self, board, player1_first=True):
        # Fill background
        self.screen.fill(self.BROWN)
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.margin + (self.board_size-1) * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos)
            
            # Horizontal lines
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.margin + (self.board_size-1) * self.cell_size, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos)
        
        # Draw stones
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board[x][y] != 0:
                    center = (
                        self.margin + y * self.cell_size,
                        self.margin + x * self.cell_size
                    )
                    # Adjust color based on player order
                    if player1_first:
                        color = self.WHITE if board[x][y] == 1 else self.BLACK
                    else:
                        color = self.BLACK if board[x][y] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, color, center, self.cell_size // 2 - 2)

    def draw_game_over(self, result, is_final_round=False):
        """Draw game over message and control buttons"""
        # Clear the bottom area first
        pygame.draw.rect(self.screen, self.BROWN, 
                        (0, self.window_size, self.window_size, self.bottom_margin))
        
        # Draw result message - moved up by adjusting the vertical position
        if result == 1:
            msg = "You Win!"
        elif result == -1:
            msg = "AI Wins!"
        else:
            msg = "Draw!"
        
        text = self.font.render(msg, True, self.RED)
        text_rect = text.get_rect(
            center=(self.window_size // 2, 
                   self.window_size + self.bottom_margin // 10)  # button and text position fixed...
        )
        self.screen.blit(text, text_rect)
        
        # Draw buttons (position unchanged)
        if not is_final_round:
            pygame.draw.rect(self.screen, self.WHITE, self.next_button)
            next_text = self.font.render("Next Game", True, self.BLACK)
            next_text_rect = next_text.get_rect(center=self.next_button.center)
            self.screen.blit(next_text, next_text_rect)
        
        pygame.draw.rect(self.screen, self.WHITE, self.quit_button)
        quit_text = self.font.render("Quit", True, self.BLACK)
        quit_text_rect = quit_text.get_rect(center=self.quit_button.center)
        self.screen.blit(quit_text, quit_text_rect)

    def handle_game_over_input(self):
        """Handle button clicks after game over"""
        mouse_pos = pygame.mouse.get_pos()
        
        if pygame.mouse.get_pressed()[0]:  # Left click
            if self.next_button.collidepoint(mouse_pos):
                return "next"
            elif self.quit_button.collidepoint(mouse_pos):
                return "quit"
        return None


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
        self.gui = GomokuGUI(game.n)

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        self.gui.draw_board(board)
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.gui.get_mouse_position()
                    if pos:
                        x, y = pos
                        # Fix: Swap x and y to match the game's internal representation
                        a = self.game.n * y + x  # Changed from n * x + y
                        if valid[a]:
                            pygame.display.flip()
                            return a
            
            pygame.display.flip()


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
        self.player1_first = True # Track original player order
        self.current_round = 0
        self.total_rounds = 0  # Will be set in playGames

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2, 0 if draw)
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        
        # Reset GUI for new game if it exists
        if hasattr(self.game, 'gui'):
            # Pass player order information to GUI
            self.game.gui = GomokuGUI(len(board), self.player1_first)
        
        it = 0
        while self.game.getGameEnded(board, curPlayer) is None:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board, self.player1_first)  # Pass player order information
            
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
        
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(curPlayer))
            self.display(board, self.player1_first)  # Pass player order information
            
            if hasattr(self.game, 'gui'):
                result = curPlayer * self.game.getGameEnded(board, curPlayer)
                if not self.player1_first:
                    result = -result
                
                # Check if this is the final round
                self.current_round += 1
                is_final_round = (self.current_round >= self.total_rounds)
                
                self.game.gui.draw_game_over(result, is_final_round)
                pygame.display.flip()
                
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    
                    action = self.game.gui.handle_game_over_input()
                    if action == "next" and not is_final_round:
                        break
                    elif action == "quit":
                        pygame.quit()
                        sys.exit()
                    
                    pygame.display.flip()
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        self.total_rounds = num
        self.current_round = 0
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        self.player1_first = True  # First half: player1 goes first
        for _ in tqdm(range(num), desc="Arena.playGames (player1 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1
        self.player1_first = False  # Second half: original player2 goes first

        for _ in tqdm(range(num), desc="Arena.playGames (player2 go first)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
