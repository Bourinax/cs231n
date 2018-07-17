import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    scores = scores - np.max(scores) # check dimensions here
    #correct_class_score = scores[y[i]]
    probs = np.exp(scores)/np.sum(np.exp(scores))
    #print(probs)
    loss -= np.log(probs[y[i]])
    for j in range(num_classes):
      #print("shape: " + str(np.shape(X[i,:])))
      b=probs[j]-int(j==y[i])
      #print(b)
      dW[:,j] += (probs[j]-int(j==y[i]))*X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores = scores - np.amax(scores, axis = 1, keepdims = 1) # check dimensions here
  probs = np.exp(scores)/np.sum(np.exp(scores), axis = 1, keepdims = 1)
  ind1 = np.indices((num_classes, num_train))[0]
  # print("shape ind1: " + str(np.shape(ind1)))
  ind2 = np.repeat(y[np.newaxis,:], num_classes, axis = 0)
  #print("shape ind2: " + str(np.shape(ind2)))
  ind = (ind1==ind2) # indicator matrix
  #print(ind)
  #print("shape ind: " + str(np.shape(ind)))
    
  #print("shape probs: " + str(np.shape(probs)))
  truc = probs - ind.T
  dW = (X.T).dot(truc) 
  #foo = probs[:,y]
  #print("shape foo: " + str(np.shape(foo)))
  loss = -np.sum(np.log(probs[np.arange(num_train),y]))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

