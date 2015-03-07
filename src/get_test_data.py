import numpy as np
import pgn
import chess
import pickle
import copy
from util import *

DATA = "../data/FICS_2000.pgn"
NUM_TRAIN_GAMES = 3000
NUM_TEST_GAMES = 1000

print "Loading PGN file..."

games = get_all_games(DATA)
games = games[NUM_TRAIN_GAMES:NUM_TRAIN_GAMES+NUM_TEST_GAMES]

print "Finished loading the PGN file."
print "Total number of games: %d" % len(games)

X_test = []

for index, game in enumerate(games):
	if index % 100 == 0:
		print "Processed %d games out of %d" % (index, NUM_TEST_GAMES)

	board = chess.Bitboard()
	moves = game.moves

	for move_index, move in enumerate(moves):
		if move[0].isalpha(): # check if move is SAN		
			if move_index % 2 == 0:
				im = convert_bitboard_to_image(board)
			else:
				im = flip_image(convert_bitboard_to_image(board))
				im = flip_color(im)

			im = np.rollaxis(im, 2, 0) # to get into form (C, H, W)

			# Filling the X_test array
			X_test.append(im)

			board.push_san(move)
		
X_test = np.array(X_test)

print "Processed %d games out of %d" % (NUM_TEST_GAMES, NUM_TEST_GAMES)
print "Saving test data..."

output = open('X_test_%d.pkl' % NUM_TEST_GAMES, 'wb')
pickle.dump(X_test, output)
output.close()

print "Done!"