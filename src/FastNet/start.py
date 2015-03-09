from run import *
from classifiers.convnet import *

init_convnet = init_chess_convnet
convnet = chess_convnet

X_train = get_data("../../data/train_data/X_train_3000.pkl")

y_train = get_data("../../data/train_data/y_train_3000.pkl")

models = initModels(init_convnet)

results = train(X_train[:1000], y_train[:1000], X_train[1000:1300], y_train[1000:1300], models['Piece'], convnet)