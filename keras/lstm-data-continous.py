# lstm model
from numpy import mean, std
from numpy import array
from numpy import dstack
from numpy import transpose
from pandas import read_csv, concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

def load_data(file):
    chunk_size = 16  # 定义每次读取的行数
    # 使用pandas的read_csv函数来逐块读取CSV文件
    total_data = read_csv(file)
    loaded = []

    # 滑动窗口读取数据
    for i in range(len(total_data) - chunk_size + 1):
        chunk = total_data.iloc[i:i+chunk_size, :20]
        loaded.append(chunk)

    return loaded

# # load train data
filepath = 'F:/Academic/Codes/Data/LSTM/k-fold/continous/'  # 您的CSV文件路径
# train_X = 'train_x.csv'
# train_Y = 'train_y.csv'
# train_file = filepath + train_X

# trainX = load_data(train_file)
# trainX = transpose(trainX, (2, 0, 1))
# trainy = read_csv(filepath + train_Y, delim_whitespace=True)
# trainy = to_categorical(trainy)

# print('train shape:',trainX.shape, trainy.shape)

# load train data
test_X = 'test_x.csv'
test_Y = 'test_y.csv'
test_file = filepath + test_X

testX = load_data(test_file)
# testX = transpose(testX, (2, 0, 1))
testX = array(testX)
testy = read_csv(filepath + test_Y, delim_whitespace=True)
print('test shape:',testX.shape, testy.shape)
testy = to_categorical(testy)


