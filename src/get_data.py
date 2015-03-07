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

all_pairs = []
for index, game in enumerate(games):
	if index % 100 == 0:
		print "Processed %d games out of %d" % (index, NUM_GAMES)

	board = chess.Bitboard()
	moves = game.moves

	previous_board = copy.deepcopy(board)
	for move_index, move in enumerate(moves):
		pair = []
		if move[0].isalpha(): # check if move is SAN
			board.push_san(move)
			if move_index % 2 == 0:
				im = convert_bitboard_to_image(board)
				previous_im = convert_bitboard_to_image(previous_board)
			else:
				im = flip_image(convert_bitboard_to_image(board))
				im = flip_color(im)
				previous_im = flip_image(convert_bitboard_to_image(previous_board))
				previous_im = flip_color(previous_im)
			
			pair.append(previous_im)
			pair.append(im)
			all_pairs.append(pair)
			
			previous_board = copy.deepcopy(board)

print "Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES)
print "Saving data..."

output = open('data_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(all_pairs, output)
output.close()

print "Done!"