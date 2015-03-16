import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
from layers import *
from fast_layers import *
from classifiers.convnet import *
from classifier_trainer import ClassifierTrainer
from gradient_check import eval_numerical_gradient_array
from util import *

def get_data(path):
	f = open(path)
	return pickle.load(f)

def initModels(fn):
	pieceModel = fn()
	pawnModel = fn()
	bishopModel = fn()
	knightModel = fn()
	queenModel = fn()
	kingModel = fn()
	rookModel = fn()
	models = {'Piece': pieceModel, 'P': pawnModel, 'B': bishopModel, 'R': rookModel, 'Q': queenModel, 'K':kingModel, 'N':knightModel}
	return models

def save_data(data, name):
	output = open(name, 'wb')
	pickle.dump(data, output)
	output.close()

def train(X_train, y_train, X_val, y_val, model, fn):
	trainer = ClassifierTrainer()
	print X_train.shape, y_train.shape, X_val.shape, y_val.shape
	best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
          	X_train, y_train, X_val, y_val, model, fn,
          	reg=0.0000, learning_rate=0.0015, batch_size=250, num_epochs=15,
          	learning_rate_decay=0.999, update='rmsprop', verbose=True, dropout=1.0)

	return (best_model, loss_history, train_acc_history, val_acc_history)

def plot(loss_history, train_acc_history, val_acc_history):
	plt.subplot(2, 1, 1)
	plt.plot(train_acc_history)
	plt.plot(val_acc_history)
	plt.title('accuracy vs time')
	plt.legend(['train', 'val'], loc=4)
	plt.xlabel('epoch')
	plt.ylabel('classification accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss_history)
	plt.title('loss vs time')
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.show()

def gradient_check(X, model, y):
	loss, grads = chess_convnet(X, model, y)
	dx_num = eval_numerical_gradient_array(lambda x: chess_convnet(x, model)[1]['W1'], x, grads)
	return rel_error(dx_num, grads['W1'])

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

def predictMove(img, models):
	modelScores = {}
	scores = three_layer_convnet([img], models['piece'])
	for key in models.keys():
		if key != 'piece':
			modelScores[key] = three_layer_convnet([img], models[key])

	availablePiecesBoard = clip_pieces(scores, img) # (1, 64) size

	maxScore = 0
	maxFromCoordinate, maxToCoordinate = None
	for i in range(64):
		coordinate = scoreToCoordinateIndex(i)
		if availablePiecesBoard[i] != 0:
			pieceType = INDEX_TO_PIECE[np.argmax(img[:, coordinate[0], coordinate[1]])]
			availableMovesBoard = clip_moves(modelScores[pieceType], img, coordinate)
			composedScore = np.max(boardToScores(availableMovesBoard)) * availablePiecesBoard[i]
			if composedScore > maxScore:
				maxScore = composedScore
				maxFromCoordinate, maxToCoordinate = coordinate, scoreToCoordinateIndex(np.argmax(boardToScores(availableMovesBoard)))

	maxFromCoordinate = coord2d_to_chess_coord(maxFromCoordinate)
	maxToCoordinate = coord2d_to_chess_coord(maxToCoordinate)

	return maxFromCoordinate + maxToCoordinate

def main():
	pass

if __name__ == "__main__":
    main()