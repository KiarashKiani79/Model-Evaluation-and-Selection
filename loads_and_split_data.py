from my_common_imports import *

def LoadAndSplitData():
    # Load the dataset
    data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',') # data.shape = (50, 2)

    # Split the data set to x_train/ x_cv/ x_test
    x = np.expand_dims(data[:,0], axis=1) # x.shape = (50,1)
    y = np.expand_dims(data[:,1], axis=1) # y.shape = (50,1)

    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1,) 
    # x_train.shape = (30, 1)
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
    # x_cv.shape = x_test.shape = (10, 1)
    del x_, y_ 
    
    return x_train, y_train, x_cv, y_cv, x_test, y_test