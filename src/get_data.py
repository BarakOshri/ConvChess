import numpy as np
import pgn
import chess
import pickle
from util import *

DATA = "../data/FICS_2000.pgn"
NUM_GAMES = 10000

games = get_all_games(DATA)
games = games[:NUM_GAMES]

print "Total number of games: %d" % len(games)

all_boards = []
for index, game in enumerate(games):
	if index % 100 == 0:
		print "Processed %d games out of %d" % (index, NUM_GAMES)

	board = chess.Bitboard()
	moves = game.moves

	for move in moves:
		if move[0].isalpha(): # check if move is SAN
			board.push_san(move)
			all_boards.append(convert_bitboard_to_image(board))

print "Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES)
print "Saving data..."

output = open('data_10000.pkl', 'wb')
pickle.dump(all_boards, output)
output.close()

print "Done!"