import pgn
import chess
import numpy as np

BOARD_SIZE = (8, 8, 6)
PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}

# Reads all the chessboard games in the provided
# pgn file. Returns games
def get_all_games(file_name):
	pgn_file = open(file_name)
	pgn_text = pgn_file.read()
	pgn_file.close()
	return pgn.loads(pgn_text)

def initialize_board():
	board = np.zeros(BOARD_SIZE)
	pattern = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']

	# Putting the black non-pawn pieces
	for index, piece in enumerate(pattern):
		board[0, index, PIECE_TO_INDEX[piece]] = -1

	# Putting the black pawns
	board[1, :, PIECE_TO_INDEX['P']] = -1

	# Putting the white non-pawn pieces
	for index, piece in enumerate(pattern):
		board[BOARD_SIZE[0] - 1, index, PIECE_TO_INDEX[piece]] = 1

	# Putting the white pawns
	board[BOARD_SIZE[0] - 2, :, PIECE_TO_INDEX['P']] = 1

	return board
	
def convert_bitboard_to_image(board):
	im2d = np.array(list(str(board).replace('\n', '').replace(' ', ''))).reshape((8, 8))
	im = np.zeros(BOARD_SIZE)

	im2d = np.array(list(str(board).replace('\n', '').replace(' ', ''))).reshape((8, 8))
	im = np.zeros(BOARD_SIZE)

	for i in xrange(BOARD_SIZE[0]):
		for j in xrange(BOARD_SIZE[1]):
			piece = im2d[i, j]
			if piece == '.': continue
			if piece.isupper():
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = 1
			else:
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = -1

	