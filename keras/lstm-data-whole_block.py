# lstm model
from numpy import mean, std
from numpy import array
from numpy import dstack
from numpy import transpose
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot

def load_data(file):
    loaded = list()
    chunk_size = 16  # 定义每次读取的行数
    # 使用pandas的read_csv函数来逐块读取CSV文件
    chunk_iterator = read_csv(file, chunksize=chunk_size)

    # 遍历每个数据块
    for chunk in chunk_iterator:
        # 在这里，'chunk' 包含了每次读取的16行数据
        # 使用切片选择前20列数据并打印
        chunk = chunk.iloc[:, :20]  # 选择前20列
        loaded.append(chunk)

    # # stack group so that features are the 3rd dimension
    # loaded = dstack(loaded)
    return loaded

# load train data
filepath = 'F:/Academic/Codes/Data/LSTM/k-fold/'  # 您的CSV文件路径
train_X = 'train_x_9.csv'
train_Y = 'train_y_9.csv'
train_file = filepath + train_X

trainX = load_data(train_file)
# trainX = transpose(trainX, (2, 0, 1))
trainX = array(trainX)
trainy = read_csv(filepath + train_Y, delim_whitespace=True)
print('train shape:',trainX.shape, trainy.shape)
trainy = to_categorical(trainy)

# load train data
test_X = 'test_x_9.csv'
test_Y = 'test_y_9.csv'
test_file = filepath + test_X

testX = load_data(test_file)
# testX = transpose(testX, (2, 0, 1))
testX = array(testX)
testy = read_csv(filepath + test_Y, delim_whitespace=True)
print('test shape:',testX.shape, testy.shape)
testy = to_categorical(testy)

print(trainX.shape, trainy.shape, testX.shape, testy.shape)

