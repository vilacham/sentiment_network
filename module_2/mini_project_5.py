import time
import sys
import numpy as np


# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """ Create a SentimentMLP with the given settings.

        Arguments
        ---------
        reviews (list): List of reviews used for training
        labels (list): List of POSITIVE/NEGATIVE labels associated with the
        given reviews
        hidden_nodes (int): Number of nodes to create in the hidden layer
        learning_rate (float): Learning rate used while training
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducible results during development
        np.random.seed(1)

        # Process the reviews and their associated labels so that everything is
        # ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning
        # rate that were passed into this initializer, the same number of input
        # nodes as there are vocabulary words and create a single output node.
        self.initialize_network(len(self.review_vocab), hidden_nodes, 1,
                                learning_rate)

    def pre_process_data(self, reviews, labels):
        """ Process the reviews and their associated labels so that everything
        is ready for training.

        Arguments
        ---------
        reviews (list): List of reviews used for training
        labels (list): List of POSITIVE/NEGATIVE labels associated with the
        given reviews
        """
        # Create empty set to store words from the reviews
        review_vocab = set()

        # Populate review vocab with all the words in the given reviews.
        for review in reviews:
            review_vocab.update(review.split(' '))

        # Convert the vocabulary set to a list so we can access words via
        # indices
        self.review_vocab = list(review_vocab)

        # Store the size of the review
        self.review_vocab_size = len(self.review_vocab)

        # Create a dictionary of words in the vocabulary mapped to index
        # positions
        self.word2index = {}

        # Populate dict with indices for all words in self.review_vocab
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

    def initialize_network(self, input_nodes, hidden_nodes, output_nodes,
                           learning_rate):
        """ Build the network to have the number of input nodes, hidden nodes,
        output nodes and learning rate passed to this function.

        Arguments
        ---------
        input_nodes (int): Number of nodes in the input layer
        hidden_nodes (int): Number of nodes in the hidden layer
        output_nodes (int): Number of nodes in the output layer
        learning_rate (float): Learning rate used while training
        """
        # Store the number of nodes in the input, hidden and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights between the input and hidden layers as a matrix of
        # zeros
        self.w_i_h = np.zeros((self.input_nodes, self.hidden_nodes))

        # Initialize weights between the hidden and output layers as a matrix
        # of random values
        self.w_h_o = np.random.normal(0.0, self.output_nodes ** -0.5,
                                      (self.hidden_nodes, self.output_nodes))

        # Initialize hidden layer as a matrix of zeros
        self.hidden_layer = np.zeros((1, self.hidden_nodes))

    def train(self, training_reviews_raw, training_labels):
        """ Train the SentimentMLP.

        Arguments
        ---------
        training_reviews_raw (list): List of reviews.
        training_labels (list): List of labels.
        """
        # Make sure we have a matching number of reviews and labels
        assert(len(training_reviews_raw) == len(training_labels))

        # Create a list of indices for each review (from self.word2index) to
        # deal with non-zero inputs
        training_reviews = list()
        for training_review in training_reviews_raw:
            indices =set()
            for word in training_review.split(' '):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Print message
        print('-' * 21 + ' TRAINING ' + '-' * 22)

        # Remember when training started for printing time statistics
        start = time.time()

        # Loop through all the given reviews and run a forward and backward
        # pass, updating weights for every item
        for i in range(len(training_reviews)):
            # Get the review and its correct label
            review, label = training_reviews[i], training_labels[i]

            # Implement forward pass through the network (no activation
            # function for the hidden layer, sigmoid activation function for
            # the output layer)
            self.hidden_layer *= 0
            for index in review:
                self.hidden_layer += self.w_i_h[index]
            final_input = np.dot(self.hidden_layer, self.w_h_o)
            final_output = sigmoid(final_input)

            # Implement backward pass through the network
            error = get_target_for_label(label) - final_output
            output_error_term = error * sigmoid_derivative(final_output)
            hidden_error = np.dot(output_error_term, self.w_h_o.T)
            hidden_error_term = hidden_error
            self.w_h_o += self.learning_rate * np.dot(self.hidden_layer.T,
                                                      output_error_term)
            for index in review:
                self.w_i_h[index] += self.learning_rate * hidden_error_term[0]

            # Keep track of correct predictions (a prediction will be correct
            # if the absolute value of the output error is less than 0.5)
            if abs(error) < 0.5:
                correct_so_far += 1

            # For debug process, print out training accuracy and speed
            # throughout the training process
            elapsed_time = float(time.time() - start)
            reviews_per_sec = i / elapsed_time if elapsed_time > 0 else 0
            progress = 100 * i/float(len(training_reviews_raw))
            training_acc = correct_so_far * 100 / float(i + 1)
            sys.stdout.write('\rProgress: {:6.2f}%'.format(progress)
                             + ' | Speed: {:6.1f}/sec'.format(reviews_per_sec)
                             + ' | Acc: {:7.3f}%'.format(training_acc))
            sys.stdout.flush()
            if progress % 10 == 0 and progress != 0:
                print("")
        print("")

    def test(self, testing_reviews, testing_labels):
        """ Test the SentimentMLP.

        Arguments
        ---------
        testing_reviews (list): List of reviews
        testing_labels (list): List of labels
        """
        # Keep track of correct predictions to display accuracy during testing
        correct_so_far = 0

        # Print message
        print('-' * 22 + ' TESTING ' + '-' * 22)

        # Remember when testing started for printing time statistics
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            # Call run to predict label of the current review
            prediction = self.run(testing_reviews[i])

            # Keep track of correct predictions
            if prediction == testing_labels[i]:
                correct_so_far += 1

            # For debug purposes, print out testing accuracy and speed
            # throughout the testing process
            elapsed_time = float(time.time() - start)
            reviews_per_sec = i / elapsed_time if elapsed_time > 0 else 0
            progress= 100 * i / float(len(testing_reviews))
            testing_acc = correct_so_far * 100 / float(i + 1)
            sys.stdout.write('\rProgress: {:6.2f}%'.format(progress)
                             + ' | Speed: {:6.1f}/sec'.format(reviews_per_sec)
                             + ' | Acc: {:7.3f}%'.format(testing_acc))
            sys.stdout.flush()
            sys.stdout.flush()
        print("")

    def run(self, review):
        """ Predict whether a review is positive or negative.

        Arguments
        ---------
        review (str): Review to be predicted as positive or negative

        Returns
        -------
        prediction (str): Prediction (positive or negative)
        """
        # Run a forward pass through the network
        self.hidden_layer *= 0
        indices = set()
        for word in review.lower().split(' '):
            if word in self.word2index.keys():
                indices.add(self.word2index[word])
        for index in indices:
            self.hidden_layer += self.w_i_h[index]
        final_output = sigmoid(np.dot(self.hidden_layer, self.w_h_o))

        # Predict as 'POSITIVE' if final_output is greater or equal 0.5 and as
        # 'NEGATIVE' otherwise
        if final_output >= 0.5:
            prediction = 'POSITIVE'
        else:
            prediction = 'NEGATIVE'

        return prediction


def get_target_for_label(label):
    """ Convert a label to zero or one.

    Arguments
    ---------
    label (str): Label ('POSITIVE' or 'NEGATIVE')

    Returns
    -------
    target (int): Label mapped to int.
    """
    if label == 'POSITIVE':
        target = 1
    else:
        target = 0
    return target


def sigmoid(x):
    """ Calculate the result of calculating the sigmoid activation
    function.

    Arguments
    ---------
    x (int, float): Value of the input

    Returns
    -------
    y (float): The sigmoid value
    """
    # Calculate and return the sigmoid of the input value
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_derivative(output):
    """ Calculate the derivative of the sigmoid activation function.

    Arguments
    ---------
    output (float): The original output from the sigmoid function

    Returns
    -------
    derivative (float): The derivative of the sigmoid function
    """
    # Calculate and return the derivative of the sigmoid function
    derivative = output * (1 - output)
    return derivative
