# Adapted from Erik Bern's Chess AI
# https://github.com/erikbern/deep-pink

import numpy as np
import sunfish
import chess
import pickle
import random
import time
import traceback
import re
import string
import math
from util import *
from run import *
from chess import pgn
from layers import *
from fast_layers import *
from classifiers.convnet import *
from classifier_trainer import ClassifierTrainer
from time import time

trained_models = {}
INDEX_TO_PIECE_2 = {0 : 'Pawn', 1 : 'R', 2 : 'N', 3 : 'B', 4 : 'Q', 5 : 'K'}

def load_models():
    model_names = ['piece', 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
    names = ['Piece', 'P', 'R', 'N', 'B', 'Q', 'K']

    for index, model_name in enumerate(model_names):
        path = '%s_model.pkl' % model_name
        trained_model = get_data(path)
        trained_models[names[index]] = trained_model

def predict(X, model, fn):
    return fn(X, model)

def predictionAccuracy(predictions, label):
    return np.mean(predictions == label)

def scoreToCoordinateIndex(score):
    return (score/8, score%8)

def scoresToBoard(scores):
    return scores.reshape((8, 8))

def boardToScores(board):
    return board.reshape((64))

def predictMove(img):
    modelScores = {}
    scores = three_layer_convnet(np.array([img]), trained_models['Piece'])
    for key in trained_models.keys():
        if key != 'Piece':
            modelScores[key] = three_layer_convnet(np.array([img]), trained_models[key])

    availablePiecesBoard = clip_pieces_single(scores, img) # (1, 64) size
    maxScore = 0
    maxFromCoordinate, maxToCoordinate = -1, -1
    availablePiecesBoard = np.reshape(availablePiecesBoard, (64))
    for i in xrange(64):
        coordinate = scoreToCoordinateIndex(i)
        if availablePiecesBoard[i] != 0:
            pieceType = INDEX_TO_PIECE[np.argmax(img[:, coordinate[0], coordinate[1]])]
            availableMovesBoard = clip_moves(modelScores[pieceType], img, coordinate)
            # print "Best move score: ", np.max(boardToScores(availableMovesBoard))
            # print "Its coordinate; ", coordinate
            composedScore = np.max(boardToScores(availableMovesBoard)) * availablePiecesBoard[i]
            if composedScore > maxScore:
                # print "I increased maxscore!"
                maxScore = composedScore
                maxFromCoordinate, maxToCoordinate = coordinate, scoreToCoordinateIndex(np.argmax(boardToScores(availableMovesBoard)))

    maxFromCoordinate = coord2d_to_chess_coord(maxFromCoordinate)
    maxToCoordinate = coord2d_to_chess_coord(maxToCoordinate)   

    return maxFromCoordinate + maxToCoordinate

def create_move(board, crdn):
    # workaround for pawn promotions
    move = chess.Move.from_uci(crdn)
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if int(move.to_square/8) in [0, 7]:
            move.promotion = chess.QUEEN # always promote to queen
    return move

class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()

class Computer(Player):
    def move(self, gn_current):
        bb = gn_current.board()

        im = convert_bitboard_to_image(bb)
        im = np.rollaxis(im, 2, 0)
        
        move_str = predictMove(im)
        move = chess.Move.from_uci(move_str)

        if move not in bb.legal_moves:
            print "NOT A LEGAL MOVE"

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move
        
        return gn_new

class Human(Player):
    def move(self, gn_current):
        bb = gn_current.board()

        #print bb

        def get_move(move_str):
            try:
                move = chess.Move.from_uci(move_str)
            except:
                print 'cant parse'
                return False
            if move not in bb.legal_moves:
                print 'not a legal move'
                return False
            else:
                return move

        while True:
            print 'your turn:'
            move = get_move(raw_input())
            if move:
                break

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        print move
        gn_new.move = move
        
        return gn_new

class Sunfish(Player):
    def __init__(self, maxn=1e4):
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxn = maxn

    def move(self, gn_current):
        import sunfish

        assert(gn_current.board().turn == 1)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)

        #t0 = time.time()
        move, score = sunfish.search(self._pos, maxn=self._maxn)
        #print time.time() - t0, move, score
        self._pos = self._pos.move(move)

        crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        move = create_move(gn_current.board(), crdn)
        
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

def game():
    gn_current = chess.pgn.Game()

    maxn = 10 ** (2.0 + random.random() * 1.0) # max nodes for sunfish

    print 'maxn %f' % (maxn)

    player_a = Computer()
    player_b = Human()
    #player_b = Sunfish(maxn=maxn)

    times = {'A' : 0.0, 'B' : 0.0}

    while True:
        for side, player in [('A', player_a), ('B', player_b)]:
            #t0 = time.time()
            try:
                gn_current = player.move(gn_current)
            except KeyboardInterrupt:
                return
            except:
                traceback.print_exc()
                return side + '-exception'

            #times[side] += time.time() - t0
            print '=========== Player %s: %s' % (side, gn_current.move)
            s = str(gn_current.board())
            print s
            if gn_current.board().is_checkmate():
                return side
            elif gn_current.board().is_stalemate():
                return '-'
            elif gn_current.board().can_claim_fifty_moves():
                return '-' 
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                return side

def play():
    while True:
        side = game()
        f = open('stats.txt', 'a')
        f.write('%s\n' % (side))
        f.close()

if __name__ == '__main__':
    load_models()
    play()
