import tensorflow as tf
import numpy as np
import chess
from tqdm import tqdm
import os


def fen_to_onehot(fen):
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    board_array = np.zeros((12, 8, 8), dtype=np.int8)

    board = chess.Board(fen)

    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, row))
            if piece is not None:
                board_array[piece_map[str(piece)], row, col] = 1
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
    return one_hot_tensor


def parse_data_line(line):
    line = ''.join([chr(char) for char in tf.strings.unicode_decode(line, input_encoding='UTF-8').numpy()])

    parts = line.rsplit(' ', 1)
    fen = parts[0]
    score = parts[1]
    tensor = fen_to_onehot(fen)
    score = float(score)
    if score > 32.0:
        score = 32.0
    if score < -32.0:
        score = -32.0
    return tensor, score


with open('training_data.txt', 'r') as f:
    datafull = f.readlines()

import random

random.shuffle(datafull)
print("data loaded")

# data = datafull[int(len(datafull)*0.9):]
data = datafull

batch_size = 1024


def create_dataset(lines):
    # Use tf.data.Dataset.from_generator to create a dataset from a generator function
    def generator():
        with tqdm(total=len(lines)) as pbar:
            for i, line in enumerate(lines):
                parsed_line = parse_data_line(line)
                yield parsed_line
                pbar.update(1)

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))
    return dataset


train_data = data[:int(len(data) * 0.8)]
val_data = data[int(len(data) * 0.8):]

print("data split")

# Create train and val datasets using the create_dataset function
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)

# Save the datasets using tf.data.Dataset.save with compression='GZIP'
tf.data.Dataset.save(train_dataset, './train_datasetv40', compression='GZIP')
tf.data.Dataset.save(val_dataset, './val_datasetv40', compression='GZIP')
