from run import *
from classifiers.convnet import *

init_convnet = init_three_layer_convnet
convnet = three_layer_convnet

# init_convnet = init_conv_convnet
# convnet = conv_convnet

numToPiece = {0: 'Piece', 1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K'}

netsToTrain = [0]

for net in netsToTrain:
	models = initModels(init_convnet)
	if net == 0:
		X = get_data("../../data/train_data/X_train_3000.pkl")
		y = get_data("../../data/train_data/y_train_3000.pkl")
		
		# X_train = X[:1000]
		# y_train = y[:1000]
		# X_val = X[1000:1100]
		# y_val = y[1000:1100]

		X_train = X[:100000]
		y_train = y[:100000]
		X_val = X[100000:110000]
		y_val = y[100000:110000]
	else:
		X = get_data("../../data/train_data/p%d_X_3000.pkl" % net)
		y = get_data("../../data/train_data/p%d_y_3000.pkl" % net)
		X_train = X[:9*X.shape[0]/10]
		y_train = y[:9*X.shape[0]/10]
		X_val = X[9*X.shape[0]/10:]
		y_val = y[9*X.shape[0]/10:]
	# print X_train.shape
	# print X_val.shape
	# print y_val.shape
	# print numToPiece[net]

	results = train(X_train, y_train, X_val, y_val, models[numToPiece[net]], convnet)

	plot(results[1], results[2], results[3])


