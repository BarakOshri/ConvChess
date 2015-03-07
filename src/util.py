import pgn
import chess
import numpy as np

BOARD_SIZE = (8, 8, 6)
PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
Y_TO_CHESSY = {0 : 'a', 1 : 'b', 2 : 'c', 3 : 'd', 4 : 'e', 5 : 'f', 6 : 'g', 7 : 'h'}
CHESSY_TO_Y = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7}

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

	for i in xrange(BOARD_SIZE[0]):
		for j in xrange(BOARD_SIZE[1]):
			piece = im2d[i, j]
			if piece == '.': continue
			if piece.isupper():
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = 1
			else:
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = -1

	return im

def flip_image(im):
	return im[::-1, :, :]

def flip_color(im):
	indices_white = np.where(im == 1)
	indices_black = np.where(im == -1)
	im[indices_white] = -1
	im[indices_black] = 1
	return im

def flip_coord2d(coord2d):
	return (8 - coord2d[0] - 1, coord2d[1])

def coord2d_to_chess_coord(coord2d):
	chess_coord = Y_TO_CHESSY[coord2d[1]] + str(8 - coord2d[0])
	return chess_coord

def chess_coord_to_coord2d(chess_coord):
	return (8 - int(chess_coord[1]), CHESSY_TO_Y[chess_coord[0]])

def flatten_coord2d(coord2d):
	return ((8 * coord2d[0]) + coord2d[1])

def clip_pieces(prob_dist, im):
	no_white_indices = np.where(im != 1)
	no_white_indices = no_white_indices[:2]
	prob_dist[no_white_indices] = 0
	return prob_dist

# def clip_move(prob_dist, im, coord):
# 	piece_coord = coord2d_to_chess_coord(coord)

# 	chess.Move.from_uci("c1c2") in board.legal_moves






	