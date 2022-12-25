#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
	array=np.ones(shape)
	return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	array=np.zeros(shape)
	return array

#input: int  
def create_identity_numpy_array(order):
	array=np.identity(order,int)
	return array

#input: numpy array
def matrix_cofactor(array):
	#return cofactor matrix of the given array
    #Formula = (A-1)T * det(A)
    inv = np.linalg.inv(array)
    trans = np.transpose(inv)
    det = np.linalg.det(array)
    cofact_mat = trans * det
    return cofact_mat

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    np.random.seed(seed1)
    W1 = np.random.rand(shape1[0],shape1[1])
    np.random.seed(seed2)
    W2 = np.random.rand(shape2[0],shape2[1])
    

    X1 = X1 ** coef1
    X2 = X2 ** coef2

    try:
        W1 = np.matmul(W1,X1)
        W2 = np.matmul(W2,X2)

        np.random.seed(seed3)
        B = np.random.rand(W1.shape[0],W1.shape[1])
        ans = W1 + W2 + B
        return ans
    
    except:
        return -1



def fill_with_mode(filename, column):
    df=pd.read_csv(filename)
    mode = df[column].mode()[0]
    df[column].fillna(mode, inplace=True)
    return df

def fill_with_group_average(df, group, column):
    df[column].fillna(df.groupby(group)[column].transform('mean'), inplace=True)
    return df


def get_rows_greater_than_avg(df, column):
    df=df[df[column] > df[column].mean()]
    return df