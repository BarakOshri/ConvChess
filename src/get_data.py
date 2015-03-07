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

move_out = []
move_in_p1, move_in_p2, move_in_p3 = [], [], []
move_in_p4, move_in_p5, move_in_p6 = [], [], []
for index, game in enumerate(games):
	if index % 100 == 0:
		print "Processed %d games out of %d" % (index, NUM_GAMES)

	board = chess.Bitboard()
	moves = game.moves

	for move_index, move in enumerate(moves):
		move_out_pair, move_in_pair = [], []
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

			board.push_san(move)

			# Filling the move_out array
			move_out_pair.append(im)
			move_out_pair.append(from_coords)
			move_out_pair = np.array(move_out_pair)

			move_out.append(move_out_pair)

			# Filling the move_in array
			move_in_pair.append(im)
			move_in_pair.append(to_coords)
			move_in_pair = np.array(move_in_pair)

			move_in_p = "move_in_p%d" % (index_piece + 1)
			move_in_p = eval(move_in_p)
			move_in_p.append(move_in_pair)
		
move_out = np.array(move_out)
move_in_p1 = np.array(move_in_p1)
move_in_p2 = np.array(move_in_p2)
move_in_p3 = np.array(move_in_p3)
move_in_p4 = np.array(move_in_p4)
move_in_p5 = np.array(move_in_p5)
move_in_p6 = np.array(move_in_p6)

print "Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES)
print "Saving data..."

output = open('move_out_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_out, output)
output.close()

output = open('move_in_p1_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p1, output)
output.close()

output = open('move_in_p2_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p2, output)
output.close()

output = open('move_in_p3_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p3, output)
output.close()

output = open('move_in_p4_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p4, output)
output.close()

output = open('move_in_p5_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p5, output)
output.close()

output = open('move_in_p6_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(move_in_p6, output)
output.close()

print "Done!"