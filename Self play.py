import chess
import chess.engine
import chess.pgn
import chess.polyglot
import tensorflow as tf
import numpy as np
import os
import time
from functools import lru_cache


def fen_to_onehot(board):
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    board_array = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = chess.square_rank(square), chess.square_file(square)
        board_array[piece_map[piece.symbol()], row, col] = 1

    # Create array of ones or zeroes depending on ptm
    ptm = board.turn
    if ptm == chess.WHITE:
        ptm_array = np.ones((8, 8), dtype=np.float32)
    else:
        ptm_array = np.zeros((8, 8), dtype=np.float32)

    attacked_bitboard = np.zeros((8, 8), dtype=np.float32)
    attacking_bitboard = np.zeros((8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square) and board.piece_at(square) != None and board.piece_at(
                square).color == chess.BLACK:
            row, col = chess.square_rank(square), chess.square_file(square)
            attacked_bitboard[row][col] = 1

        if board.is_attacked_by(chess.BLACK, square) and board.piece_at(square) != None and board.piece_at(
                square).color == chess.WHITE:
            row, col = chess.square_rank(square), chess.square_file(square)
            attacking_bitboard[row][col] = 1

    board_array = np.concatenate([board_array, [ptm_array]])
    board_array = np.concatenate([board_array, [attacked_bitboard]])
    board_array = np.concatenate([board_array, [attacking_bitboard]])
    one_hot_tensor = tf.convert_to_tensor(board_array)
    one_hot_tensor = tf.expand_dims(one_hot_tensor, axis=0)
    return one_hot_tensor


def alpha_beta(board, depth, alpha, beta, maximizing_player, transposition_table):
    global nodes_visited

    # Check transposition table for existing entry
    if transposition_table is not None and chess.polyglot.zobrist_hash(board) in transposition_table:
        entry = transposition_table[chess.polyglot.zobrist_hash(board)]
        if entry["depth"] >= depth:
            if entry["type"] == "exact":
                return entry["value"], entry["best_move"]
            elif entry["type"] == "lower_bound":
                alpha = max(alpha, entry["value"])
            elif entry["type"] == "upper_bound":
                beta = min(beta, entry["value"])
            if alpha >= beta:
                return entry["value"], entry["best_move"]

    # Base case
    if depth == 0 or board.is_game_over():
        result = evaluate(board), None

    # Maximizer's turn
    elif maximizing_player:
        max_eval = -np.inf
        best_move = None

        legal_moves = list(chess.LegalMoveGenerator(board))

        legal_moves.sort(key=lambda move: heuristic(board, move))

        for move in legal_moves:
            board.push(move)
            nodes_visited += 1
            tuple = alpha_beta(board, depth - 1, alpha, beta, False, transposition_table)
            eval = tuple[0]
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if alpha >= beta:
                break

        # Update transposition table
        if transposition_table is not None:
            if max_eval <= alpha:
                entry_type = "upper_bound"
            elif max_eval >= beta:
                entry_type = "lower_bound"
            else:
                entry_type = "exact"
            transposition_table[chess.polyglot.zobrist_hash(board)] = {"type": entry_type, "value": max_eval,
                                                                       "depth": depth, "best_move": best_move}

        result = max_eval, best_move

    # Minimizer's turn
    else:
        min_eval = np.inf
        best_move = None
        legal_moves = list(chess.LegalMoveGenerator(board))
        legal_moves.sort(key=lambda move: heuristic(board, move))
        for move in legal_moves:
            board.push(move)
            nodes_visited += 1
            eval = alpha_beta(board, depth - 1, alpha, beta, True, transposition_table)[0]
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if alpha >= beta:
                break

        # Update transposition table
        if transposition_table is not None:
            if min_eval <= alpha:
                entry_type = "upper_bound"
            elif min_eval >= beta:
                entry_type = "lower_bound"
            else:
                entry_type = "exact"
            transposition_table[chess.polyglot.zobrist_hash(board)] = {"type": entry_type, "value": min_eval,
                                                                       "depth": depth, "best_move": best_move}

        result = min_eval, best_move,

    return result


eval_counter = 0


def evaluate(board):
    global eval_counter
    eval_counter += 1
    if board.is_checkmate():
        winner = board.outcome().winner
        if winner == chess.WHITE:
            return 1000.0
        elif winner == chess.BLACK:
            return -1000.0
    else:

        tensor = fen_to_onehot(board)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        eval = interpreter.get_tensor(output_details[0]['index'])

        return eval


def heuristic(board, move):
    score = 0
    # prioritize captures
    if board.is_capture(move):
        score += 100
    # prioritize checks
    if board.gives_check(move):
        score += 50
    # prioritize moves that control the center of the board
    if move.to_square in [27, 28, 35, 36]:
        score += 10
    return score


# def alpha_beta_player(board):
#   return alpha_beta(board, depth=3, alpha=-np.inf, beta=np.inf, maximizing_player=True)[1]


engine = chess.engine
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = tf.keras.models.load_model('test 44.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

nodes_visited = 0
human_color = input("Do you want to play as white or black? Type 'w' for white or 'b' for black: ")

# Initialize the board and set the player to move
# fen = '3N4/KPP1p3/3k4/4R3/3P4/6R1/7B/8 w - - 1 1'
# board = chess.Board(fen)
board = chess.Board()
pmax = False
depth = 3
nodes_visited = 0

# Define the function that handles the computer's move
def handle_computer_move():
    global pmax, nodes_visited
    start = time.time()
    best_move = alpha_beta(board, depth=depth, alpha=-float("inf"), beta=float("inf"), maximizing_player=pmax, transposition_table={})[1]
    end = time.time()
    duration = end - start
    board.push(best_move)
    nodes_per_sec = nodes_visited / duration
    print(f"Best move: {best_move}, Nodes visited: {nodes_visited}, Current eval: {evaluate(board)}, Time taken: {duration:.2f} seconds, Nodes per second: {nodes_per_sec:.2f}")
    return best_move

# Loop until the game is over
pgn = chess.pgn.Game()
pgn.headers["Event"] = "Computer vs Computer"
while not board.is_game_over():
    # Print the board to the console
    print("\n" + str(board))
    # Handle the current player's move
    if board.turn == chess.WHITE:
        pmax = True
        move = handle_computer_move()
    else:
        pmax = False
        move = handle_computer_move()
    pgn.add_variation(move)
    nodes_visited = 0

# Print the final board and the game result
print("\n" + str(board))
result = board.result()
pgn.headers["Result"] = result
print(result)
print(pgn) # Print the entire game in PGN format