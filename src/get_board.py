import pgn
import numpy as np

BOARD_SIZE = (8, 8, 12)
WHITE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
BLACK_TO_INDEX = {'P' : 6, 'R' : 7, 'N' : 8, 'B' : 9, 'Q' : 10, 'K' : 11}

# Reads all the chess games in the provided
# pgn file. Returns games
def read_pgn_file(file_name):
	pgn_file = open(file_name)
	pgn_text = pgn_file.read()
	pgn_file.close()
	return pgn.loads(pgn_text)	

def initialize_board():
	board = np.zeros(BOARD_SIZE)
	pattern = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']

	# Putting the black non-pawn pieces
	for index, piece in enumerate(pattern):
		board[0, index, BLACK_TO_INDEX[pattern[index]]] = 1

	# Putting the black pawns
	board[1, :, BLACK_TO_INDEX['P']] = 1

	# Putting the white non-pawn pieces
	for index, piece in enumerate(pattern):
		board[0, index, WHITE_TO_INDEX[pattern[index]]] = 1

	# Putting the white pawns
	board[1, :, WHITE_TO_INDEX['P']] = 1

	return board