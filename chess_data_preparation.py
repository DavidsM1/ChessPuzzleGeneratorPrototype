import chess.pgn
import chess.svg
import chess.engine as ce
import pickle
import numpy as np
import random
import math
#import tensorflow as tf


def prepare_chess_data(filename):
    mate_in_1_positions = []
    mate_in_1_solutions = []
    engine = chess.engine.SimpleEngine.popen_uci("chess_data/stockfish_20011801_x64.exe")
    pgn = open(filename)
    game = chess.pgn.read_game(pgn)
    game_number = 1
    while game is not None:
    #for i in range(10):
        board = game.board()
        info = engine.analyse(board, chess.engine.Limit(depth=25, time=50), info=ce.INFO_ALL)
        if info['score'] != ce.PovScore(ce.Mate(2), chess.WHITE):
            print(game)
            game = chess.pgn.read_game(pgn)
            continue
        first_move = info['pv'][0]
        board.push(first_move)
        for opponent_move in board.legal_moves:
            board.push(opponent_move)
            info2 = engine.analyse(board, chess.engine.Limit(depth=25, time=50), info=ce.INFO_ALL)
            if info2['score'] == ce.PovScore(ce.Mate(1), chess.WHITE):
                mate_in_1_positions.append(board.fen())
                board.push(info2['pv'][0])
                mate_in_1_solutions.append(board.fen())
                board.pop()
            board.pop()
        if game_number%100==0:
            print("{}. games processed. {} puzzles obtained so far!".format(game_number,len(mate_in_1_positions)))
        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError:
            print(game_number+1)
        game_number = game_number + 1
    engine.quit()
    return mate_in_1_positions, mate_in_1_solutions

def save_array_to_file(filename, array):
    with open(filename, 'wb') as fp:
        pickle.dump(array, fp)

PIECE_TO_NUM_MAP = {(chess.WHITE,chess.KING):12,
                        (chess.WHITE,chess.QUEEN):1,
                        (chess.WHITE,chess.ROOK):2,
                        (chess.WHITE,chess.BISHOP):3,
                        (chess.WHITE,chess.KNIGHT):4,
                        (chess.WHITE,chess.PAWN):5,
                        (chess.BLACK, chess.KING): 6,
                        (chess.BLACK, chess.QUEEN): 7,
                        (chess.BLACK, chess.ROOK): 8,
                        (chess.BLACK, chess.BISHOP): 9,
                        (chess.BLACK, chess.KNIGHT): 10,
                        (chess.BLACK, chess.PAWN): 11}

NUM_TO_PIECE_MAP = {12:(chess.WHITE,chess.KING),
                        1:(chess.WHITE,chess.QUEEN),
                        2:(chess.WHITE,chess.ROOK),
                        3:(chess.WHITE,chess.BISHOP),
                        4:(chess.WHITE,chess.KNIGHT),
                        5:(chess.WHITE,chess.PAWN),
                        6:(chess.BLACK, chess.KING),
                        7:(chess.BLACK, chess.QUEEN),
                        8:(chess.BLACK, chess.ROOK),
                        9:(chess.BLACK, chess.BISHOP),
                        10:(chess.BLACK, chess.KNIGHT),
                        11:(chess.BLACK, chess.PAWN)}
FROM_SQUARE = 1
TO_SQUARE = 2

def prepare_NN_data(posfile):
    with open(posfile, 'rb') as fp:
        problem_set = pickle.load(fp)
    n = 1
    encoded_positions = []
    for problem in problem_set:
        board = chess.Board(problem)
        nn_pos = np.zeros((64,1), dtype=int)
        for sq_num in range(64):
            piece = board.piece_map().get(sq_num)
            if piece is not None:
                nn_pos[sq_num][0] = PIECE_TO_NUM_MAP[(piece.color,piece.piece_type)]
        nn_pos = np.reshape(nn_pos, (8,8,1))
        encoded_positions.append(nn_pos)
        if n%1000==0:
            print("{} positions encoded".format(n))
        n = n + 1

    return encoded_positions


def merge_problems_solutions():
    with open('chess_data/positions_1move.nn', 'rb') as fp:
        problem_set = pickle.load(fp)
    with open('chess_data/solutions_1move.nn', 'rb') as fp:
        solution_set = pickle.load(fp)
    problem_solution_pairs = []
    for i in range(len(problem_set)):
        problem_solution_pairs.append((problem_set[i],solution_set[i]))
        if (i+1)%10000==0:
            print("{} problem/solution pairs created".format(i+1))
    save_array_to_file('chess_data/all_pairs_1move.nn',problem_solution_pairs)


def create_ambtrain_eval_data():
    with open('chess_data/amb_problems.nn', 'rb') as fp:
        amb_problems_set = pickle.load(fp)
        print('amb problem set loaded. Size = {}'.format(len(amb_problems_set)))
    with open('chess_data/amb_solutions.nn', 'rb') as fp:
        amb_solutions_set = pickle.load(fp)
        print('amb solutions set loaded. Size = {}'.format(len(amb_solutions_set)))
    with open('chess_data/single_problems.nn', 'rb') as fp:
        det_problems_set = pickle.load(fp)
        print('deterministic problem set loaded. Size = {}'.format(len(det_problems_set)))
    with open('chess_data/single_solutions.nn', 'rb') as fp:
        det_solutions_set = pickle.load(fp)
        print('det solutions set loaded. Size = {}'.format(len(det_solutions_set)))

    print('Creating all pairs ...')
    amb_problems_solutions_pairs = []
    for i in range(len(amb_problems_set)):
        amb_problems_solutions_pairs.append((amb_problems_set[i],amb_solutions_set[i]))
        if (i+1)%10000==0:
            print("{} amb problem/solution pairs created".format(i+1))

    det_problems_solutions_pairs = []
    for i in range(len(det_problems_set)):
        det_problems_solutions_pairs.append((det_problems_set[i], det_solutions_set[i]))
        if (i + 1) % 10000 == 0:
            print("{} det problem/solution pairs created".format(i + 1))

    print('Shuffling ...')
    random.shuffle(amb_problems_solutions_pairs)
    random.shuffle(det_problems_solutions_pairs)

    print('Creating train and eval data sets ...')
    det_train_set = det_problems_solutions_pairs[:1710000]
    eval_set = det_problems_solutions_pairs[1710000:]

    train_set = amb_problems_solutions_pairs + det_train_set
    random.shuffle(train_set)
    print('Sets created - train={}, eval={}'.format(len(train_set),len(eval_set)))
    print('Saving eval set ...')
    save_array_to_file('chess_data/test_set_1move_deterministic2.nn', eval_set)
    print('Saving train set')
    save_array_to_file('chess_data/train_set_1move_amb2.nn', train_set)

#create_ambtrain_eval_data()

def test_dataSets():
    with open('chess_data/test_set_1move_deterministic2.nn', 'rb') as fp:
        test_set = pickle.load(fp)
    with open('chess_data/train_set_1move_amb2.nn', 'rb') as fp:
        train_set = pickle.load(fp)
    print("testset: {}, train_set: {}".format(len(test_set), len(train_set)))

    for i in range(700000,700100):
        print_my_problem_from_set(train_set, num=i, prefix='train')
    #solutions = [solution_to_pair(train_set[N][1])]

def print_my_problem(problem, prefix, name):
    board = convert_problem_to_board(problem)
    solutions = get_solutions(board, engine=chess_engine)
    arrows = []
    for solution in solutions:
        arrows.append(chess.svg.Arrow(solution[0], solution[1]))
    print_board_to_svg_file(board,'C:\\Users\\agris\\Desktop\\DeepAlgo\\chessfiles\\img\\chess_image_{}_{}.svg'.format(prefix,name),arrows)


def print_my_problem_from_set(set, num, prefix):
    print_my_problem(set[num][0], prefix=prefix, name=num)

def get_solutions(board, engine):
    # Solution Search
    solutions = []
    infos = engine.analyse(board, chess.engine.Limit(depth=5, time=20), info=ce.INFO_ALL, multipv=5)
    for info in infos:
        if info['score'] == ce.PovScore(ce.Mate(1), chess.WHITE):
            move = info['pv'][0]
            solutions.append((move.from_square,move.to_square))
    return solutions


def print_board_to_svg_file(board, filename, arrows=None):
    svg_image = chess.svg.board(board=board, arrows=arrows)
    with open(filename, 'w') as fp:
        fp.write(svg_image)

def convert_problem_to_board(problem, solutions=None):
    board = chess.Board()
    board.clear()
    arrows = []
    flat_problem = np.reshape(problem, (64))
    for idx,sq in enumerate(flat_problem):
        if sq != 0:
            piece = chess.Piece(NUM_TO_PIECE_MAP[sq][1],NUM_TO_PIECE_MAP[sq][0])
            board.set_piece_at(idx, piece)
    return board


def solution_to_pair(solution):
    flat_solution = np.reshape(solution, (64))
    for idx, sq in enumerate(flat_solution):
        if sq == 1:
            from_sq = idx
        if sq == 2:
            to_sq = idx
    return (from_sq, to_sq)


chess_engine = chess.engine.SimpleEngine.popen_uci("chess_data/stockfish_20011801_x64.exe")
test_dataSets()
chess_engine.quit()

def create_train_eval_data():
    with open('chess_data/all_pairs_1move.nn', 'rb') as fp:
        problem_set = pickle.load(fp)
    random.shuffle(problem_set)
    r = math.floor(len(problem_set) * 0.8)
    train_set = problem_set[:r]
    test_set = problem_set[r:]
    print(len(train_set))
    print(len(test_set))
    save_array_to_file('chess_data/train_set_1move.nn',train_set)
    save_array_to_file('chess_data/test_set_1move.nn',test_set)


def create_one_move_solutions(fen_file):
    engine = chess.engine.SimpleEngine.popen_uci("chess_data/stockfish_20011801_x64.exe")
    with open(fen_file, 'rb') as fp:
        problem_set = pickle.load(fp)
    n = 1
    encoded_ambiguous_problems = []
    encoded_ambiguous_solutions = []

    encoded_single_solution_problems = []
    encoded_single_solution_solutions = []
    
    solution_number_dict = {1:0,2:0}
    for problem in problem_set:
        board = chess.Board(problem)

        # Problem Encoding
        nn_pos = np.zeros((64,1), dtype=int)
        for sq_num in range(64):
            piece = board.piece_map().get(sq_num)
            if piece is not None:
                nn_pos[sq_num][0] = PIECE_TO_NUM_MAP[(piece.color,piece.piece_type)]
        nn_pos = np.reshape(nn_pos, (8,8))

        # Solution Search
        info = engine.analyse(board, chess.engine.Limit(depth=5, time=20), info=ce.INFO_ALL, multipv=2)
        first_move = info[0]['pv'][0]
        #print(board)
        #for move in info:
        #    print(move)
        # Solution Encoding
        nn_sol = np.zeros((64, 1), dtype=int)
        nn_sol[first_move.from_square] = FROM_SQUARE
        nn_sol[first_move.to_square] = TO_SQUARE
        nn_sol = np.reshape(nn_sol, (8, 8))

        # now - determine if the second solution is also Mate in 1 !!
        if len(info)>1 and info[1]['score']==ce.PovScore(ce.Mate(1), chess.WHITE):
            encoded_ambiguous_problems.append(nn_pos)
            encoded_ambiguous_solutions.append(nn_sol)
            sol_count = 2
            for sol in info[2:]:
                if sol['score'] == ce.PovScore(ce.Mate(1), chess.WHITE):
                    sol_count = sol_count + 1
                else:
                    break
            solution_number_dict[sol_count] = solution_number_dict[sol_count] + 1
            #print("More than one solution!")
        else:
            if len(info)==1:
                print(board)
            encoded_single_solution_problems.append(nn_pos)
            encoded_single_solution_solutions.append(nn_sol)
            solution_number_dict[1] = solution_number_dict[1] + 1
            #print("One solution!")

        if n%1000==0:
            print("{} positions encoded".format(n))
            for k, v in solution_number_dict.items():
                print("{}:{}".format(k,v))
        n = n + 1
    print("ambigous: {} / single: {}".format(len(encoded_ambiguous_problems),len(encoded_single_solution_problems)))
    engine.quit()
    return encoded_single_solution_problems, encoded_single_solution_solutions, encoded_ambiguous_problems, encoded_ambiguous_solutions

#ssp, sss, ap, ass = create_one_move_solutions('chess_data/unique_positions.fens')
#save_array_to_file('chess_data/single_problems.nn', ssp)
#save_array_to_file('chess_data/single_solutions.nn', sss)
#save_array_to_file('chess_data/amb_problems.nn', ap)
#save_array_to_file('chess_data/amb_solutions.nn',ass)

def test_dataset():
    with open('chess_data/positions.fens', 'rb') as fp:
        positions = pickle.load(fp)
    unique_fens = []
    fen_counter = {}
    for pos in positions:
        pos_value = fen_counter.get(pos, 0)
        if pos_value != 0:
            fen_counter[pos] = fen_counter[pos] + 1
        else:
            fen_counter[pos] = 1
            unique_fens.append(pos)

    total_doubles = 0
    for key, value in fen_counter.items():
        if value>1:
            total_doubles = total_doubles + value - 1
            print("{}: {}".format(key, value))
    print("Total doubles: {} / {} ({}%)==> {}".format(total_doubles, len(positions), total_doubles * 100 / len(positions),  len(unique_fens)))

    #save_array_to_file('chess_data/unique_positions.fens', unique_fens)
#test_dataset()
#prob, sol = create_one_move_solutions('chess_data\positions.fens')

#with open('chess_data/test_set.nn', 'rb') as fp:
#    set = pickle.load(fp)
#merge_problems_solutions()
#create_train_eval_data()
#print(set[0])

#nparr = np.array(set[0])
#print(np.array(set).shape)
#set2 = np.squeeze(set, axis=4)
#print(set2.shape)
#print(set2[0])
#save_array_to_file('chess_data/positions_1move.nn',prob)
#save_array_to_file('chess_data/solutions_1move.nn',sol)
