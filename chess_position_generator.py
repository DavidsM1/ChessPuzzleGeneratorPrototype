import sys
import chess
import chess.pgn
import chess.engine as ce
import time
from chess.engine import Cp, Mate, MateGiven

file = ("chess_data/1950_games.pgn")
engine = chess.engine.SimpleEngine.popen_uci("/Users/davids.miezaks/Downloads/stockfish-11-mac/Mac/stockfish-11-64")

mate_in_two_positions = []

pgn = open(file)
game = chess.pgn.read_game(pgn)
game_number = 1
with open('chess_data/generated_positions_2.pgn', 'w') as f:
    while game is not None:
        board = game.board()
        calculation_start = int(game.headers["PlyCount"]) - 6
        move_counter = 1
        for move in game.mainline_moves():
            board.push(move)
            if move_counter >= calculation_start:
                print("analyzing..")
                info = engine.analyse(board, chess.engine.Limit(time=1), info=ce.INFO_ALL)
                time.sleep(1)
                if info['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                    f.write(board.fen())
                    f.write('\n')
                    print(board.fen())
                    break
            move_counter += 1
        print("next game...")
        game = chess.pgn.read_game(pgn)
        game_number += 1
        if game_number % 10 == 0:
            print("went through ", game_number, " games")

engine.quit()
