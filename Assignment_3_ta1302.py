import pandas as pd
import numpy as np

def main():

    # Read the input file
    df = pd.read_table('./PerceptronDataF17.txt', sep= "\t", 
                       header = None, 
                       names = ['x1', 'x2', 'x3', 'x4', 'y_true'], 
                       index_col = False)
    
    print df.head()
    
    # Create the input and output matrices
    input_matrix = df.iloc[:,:4].as_matrix()
    output_matrix = df.iloc[:,4:].as_matrix()

    # Initialize the weights and bias as zero matrices
    w_old = np.zeros((4,1),dtype=int)
    b_old = np.zeros((1,1),dtype=int)

    # Find weights and bias using Perceptron Rule rule
    w_new = w_old
    b_new = b_old
    alpha = 1
    
    Q = 0
    iteration = 0
    while Q < 1000:
        temp_list = []
        for i in range(len(df)):  
            predicted_op = b_new[0][0] + np.dot(input_matrix[i].reshape(1,4), w_new)
            activated_op = activation(predicted_op)
            if activated_op != output_matrix[i][0]:
                Q = 0
                w_new = w_new + alpha * output_matrix[i][0] * input_matrix[i].reshape(4,1)
                b_new = b_new + alpha * output_matrix[i][0]
            else:
                temp_list.append(activated_op)
                Q += 1
        iteration += 1
        predicted_op_list = temp_list
    
    
    
    y_predicted = np.array(predicted_op_list).reshape(1000,1)
    df['y_pred'] = y_predicted
    
    print df
    
    print "\nIteration:", iteration, "\n"

    print "Weights New"
    print "W0:", w_new[0][0]
    print "W1:", w_new[1][0]
    print "W2:", w_new[2][0]
    print "W3:", w_new[3][0]
    print "\nBias New\n", b_new[0][0]

    #Calculate the accuracy
    accuracy = (sum(df['y_true'] == df['y_pred'])*100.0)/len(df)

    print "\nAccuracy:", accuracy

def activation(y_in):
    theta = 0.2
    if y_in > theta:
        return 1
    elif -theta <= y_in <= theta:
        return 0
    elif y_in < -theta:
        return -1
    
if __name__ == "__main__":
    main()
