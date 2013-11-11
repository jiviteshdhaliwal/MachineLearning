#####################################
# Author:   Jivitesh Singh Dhaliwal
# Date:     11-09-2013
# Program:  Perceptron Learning Algo
#####################################


###################
# Import Libraries
###################

from pylab import *                             # Pylab: Numpy+Scipy+matplotlib


###################
# Import Functions
###################

from functions import *


########
# main
########

def __main__():
    '''Perfrom machine learning using the perceptron algorithm'''

    ###############
    # Obtain Data
    ###############
    dataFileName = raw_input('Please enter the filename of data file: ')
    dataDelimiter    = raw_input('Please enter the delimiter : ')
    
    try:
        data = genfromtxt(open(dataFileName, 'r') ,delimiter = dataDelimiter, comments= '""') # Directly imports to 'data' 
        X = data[:,0:-1]                            # Assuming that the last row of data is the target value
                                                    # and all others are those of X
        y = data[:,[-1]]                            # Note the single parenthesis to assert that we are choosing
                                                    # one single column
#        if shape(X)[1] <= 2:
#            plotdata(X,y)                           # Display data to user if possible

        X = hstack((ones((shape(X)[0], 1)), X))                  # Add X0 = 1 

    except:
        print 'Unable to process data'
        exit(1)                                     # Exit the program with error code 1

    ###############################
    # Assign Theta, alpha, num_iter
    ###############################  
    
    initialWeights = rand(shape(X)[1],1)               # The number of columns of X is the no. of parameters
    alpha = float(raw_input('Enter value of alpha: ')) # Allow user to choose alpha
    numIter = int(raw_input('Iterations: '))           # User specified number of iterations
    
    ###########################
    # Perform Machine Learning
    ###########################
    
    weights = perceptron(initialWeights, X, y, alpha, numIter) 
    print 'Weights observed:', weights 
    print 'Accuracy observed: ', mean(where(h(weights,X) > 0, 1, 0) == y) * 100

    exit(0)



################
# Run Program
################

if __name__ == '__main__':
    __main__()
    exit(0)
