import numpy as np 
import matplotlib.pyplot as plt
import caffe
import argparse

def getArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("data", required=True)
	parser.add_argument("training", type=bool, required=True)
	parser.add_argument("labels", required=False)
	parser.add_argument("net", choices=['PIECE','P', 'R', 'B', 'N', 'Q', 'K'])
	parser.add_argument("gpu", type=bool)
	return parser.parse_args()

args = getArgs()
data = np.load(args.data)

if args.training: # Training

	print("Training on %d inputs." % inputs.shape[0])

	labels = np.load(args.labels)
	solver = caffe.SGDSolver('%s_solver.prototxt' % args.net)
	solver.net.set_input_arrays(data, labels)
	solver.solve()

	print ("Training complete")

else: # Testing

	print("Testing on %d inputs." % inputs.shape[0])

	classifier = caffe.Classifier("move.protxt", "%s_train.caffemodel" % args.net, gpu=args.gpu)
	prediction = classifier.predict(data)

	if args.labels:
		print ("Accuracy is %f" np.mean(prediction == labels))
	print prediction