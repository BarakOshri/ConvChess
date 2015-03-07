import numpy as np
import pgn
import chess
import pickle
import copy
from util import *

DATA = "../data/FICS_2000.pgn"
NUM_GAMES = 3000

print "Loading PGN file..."

games = get_all_games(DATA)
games = games[:NUM_GAMES]

print "Finished loading the PGN file."
print "Total number of games: %d" % len(games)

# Move-out
X_train, y_train = [], []
# Move-in
p1_X, p2_X, p3_X = [], [], []
p4_X, p5_X, p6_X = [], [], []
p1_y, p2_y, p3_y = [], [], []
p4_y, p5_y, p6_y = [], [], []
for index, game in enumerate(games):
	if index % 100 == 0:
		print "Processed %d games out of %d" % (index, NUM_GAMES)

	board = chess.Bitboard()
	moves = game.moves

	for move_index, move in enumerate(moves):
		if move[0].isalpha(): # check if move is SAN		
				
			from_to_chess_coords = board.parse_san(move)
			from_to_chess_coords = str(from_to_chess_coords)

			from_chess_coords = from_to_chess_coords[:2]
			to_chess_coords = from_to_chess_coords[2:4]
			from_coords = chess_coord_to_coord2d(from_chess_coords)
			to_coords = chess_coord_to_coord2d(to_chess_coords)
						
			if move_index % 2 == 0:
				im = convert_bitboard_to_image(board)
			else:
				im = flip_image(convert_bitboard_to_image(board))
				im = flip_color(im)
				from_coords = flip_coord2d(from_coords)
				to_coords = flip_coord2d(to_coords)

			index_piece = np.where(im[from_coords] != 0)
			# index_piece denotes the index in PIECE_TO_INDEX
			index_piece = index_piece[0][0] # ranges from 0 to 5

			from_coords = flatten_coord2d(from_coords)
			to_coords = flatten_coord2d(to_coords)

			im = np.rollaxis(im, 2, 0) # to get into form (C, H, W)

			board.push_san(move)

			# Filling the X_train and y_train array
			X_train.append(im)
			y_train.append(from_coords)

			# Filling the p_X and p_y array
			p_X = "p%d_X" % (index_piece + 1)
			p_X = eval(p_X)
			p_X.append(im)
			
			p_y = "p%d_y" % (index_piece + 1)
			p_y = eval(p_y)
			p_y.append(to_coords)
		
# Move-out
X_train, y_train = np.array(X_train), np.array(y_train)
# Move-in
p1_X, p2_X, p3_X = np.array(p1_X), np.array(p2_X), np.array(p3_X)
p4_X, p5_X, p6_X = np.array(p4_X), np.array(p5_X), np.array(p6_X)
p1_y, p2_y, p3_y = np.array(p1_y), np.array(p2_y), np.array(p3_y)
p4_y, p5_y, p6_y = np.array(p4_y), np.array(p5_y), np.array(p6_y)

print "Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES)
print "Saving data..."

print "Saving X_train array..."
output = open('X_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(X_train, output)
output.close()

print "Saving y_train array..."
output = open('y_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(y_train, output)
output.close()

for i in xrange(6):
	output_array = "p%d_X" % (i + 1)
	print "Saving %s array..." % output_array
	output_array = eval(output_array)
	output = open('p%d_X_%d.pkl' % (i + 1, NUM_GAMES), 'wb') 
	pickle.dump(output_array, output)
	output.close()

	output_array = "p%d_y" % (i + 1)
	print "Saving %s array..." % output_array
	output_array = eval(output_array)
	output = open('p%d_y_%d.pkl' % (i + 1, NUM_GAMES), 'wb') 
	pickle.dump(output_array, output)
	output.close()

print "Done!"