import pandas as pd
import numpy as np

def main():

    # Read the input file
    df = pd.read_table('./PerceptronDataF17.txt', sep= "\t", 
                       header = None, 
                       names = ['x1', 'x2', 'x3', 'x4', 'y_true'], 
                       index_col = False)

    # Create the input and output matrices
    input_matrix = df.iloc[:,:4].as_matrix()
    output_matrix = df.iloc[:,4:].as_matrix()

    # Initialize the weights and bias as zero matrices
    w_old = np.zeros((4,1),dtype=int)
    b_old = np.zeros((1,1),dtype=int)

    # Find weights and bias using Hebb rule
    w_new = w_old
    b_new = b_old
    for i in range(len(df)):
        w_new = w_new + input_matrix[i].reshape(4,1) * output_matrix[i][0]
        b_new = b_new + output_matrix[i]

    print "Weights New"
    print "W0:", w_new[0][0]
    print "W1:", w_new[1][0]
    print "W2:", w_new[2][0]
    print "W3:", w_new[3][0]
    print "\nBias New\n", b_new[0][0]

    # Computing output using new weight and bias
    temp_list = []
    for i in range(1000):
        predicted_op = b_new[0][0] + np.dot(input_matrix[i].reshape(1,4), w_new)
        temp_list.append(predicted_op[0][0])    

    # Activation Function    
    activation_func = [1 if x >=0 else -1 for x in temp_list]    

    y_predicted = np.array(activation_func).reshape(1000,1)
    df['y_pred'] = y_predicted

    #Calculate the accuracy
    accuracy = (sum(df['y_true'] == df['y_pred'])*100.0)/len(df)

    print "\nAccuracy:", accuracy
    
if __name__ == "__main__":
    main()



