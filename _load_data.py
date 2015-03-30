import os
import gzip
import cPickle
import time


import pandas as pd
import numpy as np
import theano



##################
## FUNCTIONS & CLASSES
##
##
def fn_T_support_shared_dataset( data_xy, borrow=True ):
	""" 
	Function that loads the dataset into Theano shared variables

	The reason we store our dataset in shared variables is to allow Theano to copy it into the 
	GPU memory (when code is run on GPU). Since copying data into the GPU is slow, copying a 
	minibatch everytime is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""
	data_x, data_y = data_xy
	shared_x = theano.shared( np.asarray( data_x, dtype=theano.config.floatX ), borrow=borrow )
	shared_y = theano.shared( np.asarray( data_y, dtype=theano.config.floatX ), borrow=borrow )
	# When storing data on the GPU it has to be stored as floats therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, theano.tensor.cast(shared_y, 'int32')
	##################


def fn_T_load_data_MNIST(dataset):

	start_time = time.time()
	print 100 * '-', '\n    *** Loading MNIST data...'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	# Create Theano shared variables (for GPU processing)
	test_set_x, test_set_y = fn_T_support_shared_dataset(test_set)
	valid_set_x, valid_set_y = fn_T_support_shared_dataset(valid_set)
	train_set_x, train_set_y = fn_T_support_shared_dataset(train_set)

	print '        Time elapsed:   {} seconds'.format( time.time()-start_time )
	print '        Training set:   {} {}'.format( train_set[0].shape, train_set[1].shape )
	print '        Validation set: {}'.format( valid_set[0].shape )
	print '        Testing set:    {}'.format( test_set[0].shape )

	# Output is a list of tuples. Each tuple is filled with an m-by-n matrix and an m-by-1 array.
	return [ (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), (28,28,1) ]
	##################
	
##################
##################




##################
## MAIN
##
##
if __name__=="__main__":

	print 'Just a module. Nothing here...'

##################
##################

