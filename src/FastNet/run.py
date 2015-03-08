import numpy as np
import matplotlib.pyplot as plt
from time import time
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.classifiers.convnet import *
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient_array
from ../util.py import *

def get_data(path, data=True, label=True):
	return ALL OF THE DATA

def initModels():
	pieceModel = init_chess_convnet()
	pawnModel = init_chess_convnet()
	bishopModel = init_chess_convnet()
	knightModel = init_chess_convnet()
	queenModel = init_chess_convnet()
	kingModel = init_chess_convnet()
	rookModel = init_chess_convnet()
	models = {'Piece': pieceModel, 'P': pawnModel, 'B': bishopModel, 'R': rookModel, 'Q': queenModel, 'K':kingModel, 'N':knightModel}
	return models

def dumpModels(models):

def loadModels():
	SOMETHING
	return models

def train(X_train, y_train, X_val, y_val, model):
	trainer = ClassifierTrainer()
	best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          	X_train, y_train, X_val, y_val, model, chess_convnet,
          	reg=0.05, learning_rate=0.00005, batch_size=50, num_epochs=15,
          	learning_rate_decay=1.0, update='rmsprop', verbose=True)

	return (best_model, loss_history, train_acc_history, val_acc_history)

def gradient_check(X, model, y):
	loss, grads = chess_convnet(X, model, y)
	dx_num = eval_numerical_gradient_array(lambda x: chess_convnet(x, model)[1]['W1'], x, grads)
	return rel_error(dx_num, grads['W1'])

def predict(X, model):
	return chess_convnet(X, model)

def predictionAccuracy(predictions, label):
	return np.mean(predictions == label)

def scoreToCoordinateIndex(score):
	return (score/8, score%8)

def scoresToBoard(scores):
	return scores.reshape((8, 8))

def boardToScores(board):
	return board.reshape((64))

def predictMove(img, models):
	modelScores = {}
	scores = chess_convnet([img], models['Piece'])
	for key in models.keys():
		if key != 'Piece':
			modelScores[key] = chess_convnet([img], models[key])

	availablePiecesBoard = clip_pieces(scoresToBoard(scores), img)

	maxScore = 0
	maxFromCoordinate, maxToCoordinate = None
	for i in range(64):
		coordinate = scoreToCoordinateIndex(i)
		if availablePiecesBoard[coordinate[0], coordinate[1]] != 0:
			pieceType = INDEX_TO_PIECE[np.argmax(img[:, coordinate[0], coordinate[1]])]
			availableMovesBoard = clip_moves(modelScores[pieceType], img, coordinate)
			composedScore = np.max(boardToScores(availableMovesBoard)) * availablePiecesBoard[coordinate[0], coordinate[1]]
			if composedScore > maxScore:
				maxScore = composedScore
				maxFromCoordinate, maxToCoordinate = coordinate, scoreToCoordinateIndex(np.argmax(boardToScores(availableMovesBoard)))

	return (maxFromCoordinate, MaxToCoordinate)

def main():
	pass

if __name__ == "__main__":
    main()