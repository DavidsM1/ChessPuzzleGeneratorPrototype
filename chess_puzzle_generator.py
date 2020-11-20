import sys
import chess
import chess.pgn
import chess.engine as ce
import time
from chess.engine import Cp, Mate, MateGiven

KNIGHT_VALUE = 3.3
PAWN_VALUE = 1
BISHOP_VALUE = 3.4
ROOK_VALUE = 5.3
QUEEN_VALUE = 9.7
KING_VALUE = 3.5

file = ("chess_data/generated_positions_final.pgn")
evaluation_file = ("chess_data/evaluated_generated_puzzles_full_info.pgn")
engine = chess.engine.SimpleEngine.popen_uci("/Users/davids.miezaks/Downloads/stockfish-11-mac/Mac/stockfish-11-64")
positions = []


def evaluate_chess_data(filename):

    with open(file) as f:
        positions = f.read().splitlines()

    with open(evaluation_file, 'w') as f:
        for position in positions:
            evaluation = 0
            board = chess.Board(position)
            board = remove_extra_pieces(board)
            f.write(board.fen())
            f.write('\n')
            print(board)

            white_material = board_material_evaluation(board, chess.WHITE)
            black_material = board_material_evaluation(board, chess.BLACK)

            board_material = (black_material - white_material * 100) / 45.2
            evaluation += board_material
            f.write("board material: " + str(board_material))
            f.write('\n')
            info = engine.analyse(board, chess.engine.Limit(time=0.5), info=ce.INFO_ALL, multipv=2)
            time.sleep(0.5)
            if info[0]['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                first_move = info[0]['pv'][0]
                second_move = info[0]['pv'][1]
                third_move = info[0]['pv'][2]
                from_square = first_move.from_square
                moved_piece = board.piece_at(from_square)

                moved_piece_eval = evaluate_moved_piece(board, from_square)
                evaluation += moved_piece_eval

                f.write("evaluate moved piece: " + str(moved_piece_eval))
                f.write('\n')

                # print("info pv: ", info[0]['pv'])

                board.push(first_move)

                pinned_pieces = evaluate_pinned_pieces(board)
                double_attack = evaluate_double_attack(board)

                f.write("pinned pieces: " + str(pinned_pieces))
                f.write('\n')
                f.write("double attack: " + str(double_attack))
                f.write('\n')

                evaluation += pinned_pieces
                evaluation += double_attack

                board.push(second_move)

                from_square2 = third_move.from_square
                moved_piece_eval2 = evaluate_moved_piece(board, from_square2)
                evaluation += moved_piece_eval2

                board.push(third_move)

                f.write("evaluate moved piece2: " + str(moved_piece_eval2))
                f.write('\n')

                white_material2 = board_material_evaluation(board, chess.WHITE)


                if white_material > white_material2:
                    f.write("white sacrifice material: 20")
                    f.write('\n')
                    evaluation += 20

                print(evaluation)
            if len(info) > 1:
                if info[1]['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                    evaluation -= 10
                    # f.write("extra mate: -10")
                    # f.write('\n')
                    print("another Mate in 2 found")

            f.write('total: ' + str(evaluation))
            f.write('\n')

    engine.quit()


def board_material_evaluation(board, color):
    # board = chess.BaseBoard()
    white_material = 0
    black_material = 0
    if color == chess.WHITE:
        white_material += len(board.pieces(chess.KING, chess.WHITE)) * KING_VALUE
        white_material += len(board.pieces(chess.QUEEN, chess.WHITE)) * QUEEN_VALUE
        white_material += len(board.pieces(chess.ROOK, chess.WHITE)) * ROOK_VALUE
        white_material += len(board.pieces(chess.BISHOP, chess.WHITE)) * BISHOP_VALUE
        white_material += len(board.pieces(chess.KNIGHT, chess.WHITE)) * KNIGHT_VALUE
        white_material += len(board.pieces(chess.PAWN, chess.WHITE)) * PAWN_VALUE
        return white_material
    if color == chess.BLACK:
        black_material += len(board.pieces(chess.KING, chess.BLACK)) * KING_VALUE
        black_material += len(board.pieces(chess.QUEEN, chess.BLACK)) * QUEEN_VALUE
        black_material += len(board.pieces(chess.ROOK, chess.BLACK)) * ROOK_VALUE
        black_material += len(board.pieces(chess.BISHOP, chess.BLACK)) * BISHOP_VALUE
        black_material += len(board.pieces(chess.KNIGHT, chess.BLACK)) * KNIGHT_VALUE
        black_material += len(board.pieces(chess.PAWN, chess.BLACK)) * PAWN_VALUE
        return black_material

    # difference = black_material - white_material
    # result = (difference * 100) / 45.2
    # return result


def evaluate_moved_piece(board, square):
    moved_piece = board.piece_at(square)
    piece_evaluation_arr = [[0, 20, 40, 60, 80, 100],
                            [0, 0, 20, 40, 60, 80],
                            [0, 0, 0, 20, 40, 60],
                            [0, 0, 0, 0, 20, 40],
                            [0, 0, 0, 0, 0, 20],
                            [0, 0, 0, 0, 0, 0]]
    piece_penalty_arr = [0, 0, 20, 25, 70, 100, 40]
    # piece_type_count = 0
    piece_types = []

    if len(board.pieces(chess.QUEEN, chess.WHITE)) > 0:
        piece_types.append(chess.QUEEN)
    if len(board.pieces(chess.ROOK, chess.WHITE)) > 0:
        piece_types.append(chess.ROOK)
    if len(board.pieces(chess.KING, chess.WHITE)) > 0:
        piece_types.append(chess.KING)
    if len(board.pieces(chess.BISHOP, chess.WHITE)) > 0:
        piece_types.append(chess.BISHOP)
    if len(board.pieces(chess.KNIGHT, chess.WHITE)) > 0:
        piece_types.append(chess.KNIGHT)
    if len(board.pieces(chess.PAWN, chess.WHITE)) > 0:
        piece_types.append(chess.PAWN)

    result = piece_evaluation_arr[piece_types.index(moved_piece.piece_type)][len(piece_types) - 1] - piece_penalty_arr[moved_piece.piece_type]

    # if moved_piece.piece_type == chess.KNIGHT:
    return result


def evaluate_pinned_pieces(board):
    squares = chess.SquareSet(chess.BB_ALL)
    result = 0
    for square in list(squares):
        if board.is_pinned(chess.BLACK, square):
            result = 20

    return result


def evaluate_double_attack(board):
    king = board.king(chess.BLACK)

    attackers = board.attackers(chess.WHITE, king)

    if len(attackers) > 1:
        return 20
    return 0


def remove_extra_pieces(board):
    print(board)
    curr_board = board
    result = board
    info = engine.analyse(curr_board, chess.engine.Limit(time=0.5), info=ce.INFO_ALL, multipv=2)
    time.sleep(0.5)
    move_list = info[0]['pv']
    squares = chess.SquareSet(chess.BB_ALL)
    squares.discard(curr_board.king(chess.WHITE))
    squares.discard(curr_board.king(chess.BLACK))
    for move in move_list:
        squares.discard(move.from_square)
        squares.discard(move.to_square)

    try:
        for square in squares:
            removed_piece = curr_board.remove_piece_at(square)
            if removed_piece is None:
                continue

            info = engine.analyse(curr_board, chess.engine.Limit(0.5), info=ce.INFO_ALL, multipv=2)
            time.sleep(0.5)

            if move_list != info[0]['pv']:
                if removed_piece.piece_type != chess.PAWN:
                    curr_board.set_piece_at(square=square, piece=removed_piece)
                    result = curr_board
                    continue
                if curr_board.piece_at(square + 8).piece_type == chess.PAWN:
                    extra_removed_piece = curr_board.remove_piece_at(square+8)
                    info2 = engine.analyse(curr_board, chess.engine.Limit(time=0.5), info=ce.INFO_ALL, multipv=2)
                    time.sleep(0.5)
                    if move_list != info2[0]['pv']:
                        curr_board.set_piece_at(square=square, piece=extra_removed_piece)
                        result = curr_board
                        continue
                    if len(info2) > 1:
                        if info2[1]['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                            curr_board.set_piece_at(square=square, piece=removed_piece)
                            result = curr_board
                            continue
            if len(info) > 1:
                if info[1]['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                    if removed_piece.piece_type != chess.PAWN:
                        curr_board.set_piece_at(square=square, piece=removed_piece)
                        result = curr_board
                        continue
                    if curr_board.piece_at(square + 8).piece_type == chess.PAWN:
                        extra_removed_piece = curr_board.remove_piece_at(square+8)
                        info2 = engine.analyse(curr_board, chess.engine.Limit(time=0.5), info=ce.INFO_ALL, multipv=2)
                        time.sleep(0.5)
                        if move_list != info2[0]['pv']:
                            curr_board.set_piece_at(square=square, piece=extra_removed_piece)
                            result = curr_board
                            continue
                        if len(info) > 1:
                            if info2[1]['score'] == ce.PovScore(ce.Mate(2), chess.WHITE):
                                curr_board.set_piece_at(square=square, piece=removed_piece)
                                result = curr_board
                                continue
        return result
    except:
        print("failed to remove pieces from the board")
    finally:
        print("returning back the original board")
        return board

evaluate_chess_data(file)
