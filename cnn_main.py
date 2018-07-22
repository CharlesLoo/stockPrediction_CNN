import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy import misc
import glob
import matplotlib.pyplot as plt
from PIL import Image
import math

#VGG
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from generate_data import *


#seed = 7
#np.random.seed(seed)
width = 5
height = 5
lrate = 0.01

def r_squared(y_true, y_hat):
    ssr = 0
    sst = 0
    e = np.subtract(y_true, y_hat)
    y_mean = np.mean(y_true)
    for item in e:
        ssr += item**2
    for item in y_true:
        sst += (item - y_mean)**2
    r2 = 1 - ssr / sst
    return (r2)

#compile model
def compile_model(model,lrate):
    sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    return model

#create model, using .add() construct model
def create_model():
    model = Sequential()
    #cov1
#    IN older keras version (<2.0)
#    model.add(Convolution2D(32, 3, 3,
#                           border_mode='valid', 
#                           input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), padding = 'same',input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    
    #cov2  
#    model.add(Convolution2D(32, 3, 3))
    model.add(Conv2D(32, (3, 3)))  
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
    #cov3
#    model.add(Convolution2D(64, 3, 3),border_mode='valid')
    model.add(Conv2D(64, (3, 3), padding = 'same')) 
    model.add(Activation('relu'))  
   
    #Cov4
#    model.add(Convolution2D(64, 3, 3))
    model.add(Conv2D(64, (3, 3)))	   
    model.add(Activation('relu'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
      
    model.add(Flatten())  
    model.add(Dense(256))  
    model.add(Activation('relu'))  
    model.add(Dropout(0.5))

    model.add(Dense(2))  
    model.add(Activation('softmax'))  

    return model


def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

return model
    
#compute the logarithmic return (like label)   
def find_returns(data):
    returns = []
    log_return = []
    for group in data:
        count = 30
        index = 1
        while count <= (len(group)-5):
            current_data = group[count-1]
            future_data = group[count+4]
            #coz high and low normally increas and decreas together, we use their mean to present the whole value
            p1 = np.mean(current_data)
            p2 = np.mean(future_data)
            #logarithmic return
            log_ret = math.log(p2/p1)
            returns.append(log_ret)
            log_return.append([index,log_ret])
#            if log_ret >= 0:
#               returns.append(1)
#            else:
#               returns.append(-1)
            count += 30
            index += 1
##    np.savetxt("log_return_index.txt", log_return_index, fmt="%s", delimiter=",") 
    return returns
    
    
def get_pixel_values():
    file_name = r'figures_v2'
    pixels = []
    for filename in glob.glob(file_name + '/*.png'):
        im = misc.imread(filename)
        pixels.append(im)
    return pixels
    
    
def convert_image():
    file_name = r'figures_v2'
    for filename in glob.glob(file_name + '/*.png'):
        img = Image.open(filename)
        img = img.convert('RGB')
        img.save(filename)
    
    
def plot_data(data):
    t = np.arange(0, 29, 1)
    file_name_number = 0
    fig = plt.figure(frameon=False, figsize=(width, height))
    for group in data:
        count = 30
        while count <= (len(group)-5):
            close = []
            high = []
            low = []
            op = []
##            vol = []
            for item in group[count-30:count]:
                close.append(item[0])
                high.append(item[1])
                low.append(item[2])
                op.append(item[3])
                #vol is much bigger, need to make it smaller to put them in the same img
#                vol.append(item[2]/1000000)
            file_name = r'fig_' + str(file_name_number)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(t, close[0:-1], 'y',t, high[0:-1], 'b', t, low[0:-1], 'g',t, op[0:-1], 'r')
            fig.savefig(r'figures_v2/' + file_name, dpi=100)
            fig.clf()
            file_name_number += 1
            count += 30
    print('Created %d files!' % file_name_number)
    
    
def load_sample_data():
    original_data = extract_data()
##    np.savetxt("original_data.txt", original_data, fmt="%s", delimiter=",") 
    splitted_data = split_data(original_data)
##    np.savetxt("splitted_data.txt", splitted_data, fmt="%s", delimiter=",") 
    useful_data = extract_useful_data(splitted_data)
##    np.savetxt("useful_data.txt", useful_data, fmt="%s", delimiter=",") 
    return useful_data


def extract_useful_data(data):
    groups = []
    for group in data:
        temp_buffer = []
        for item in group:
            temp = [item[1],item[2], item[3],item[4]]
            temp = [float(i) for i in temp]
            temp_buffer.append(temp)
        groups.append(temp_buffer)
    return groups


def split_data(data):
    groups = []
    for item in data:
        temp_buffer = []
        for string in item:
            number = string.split(',')
            temp_buffer.append(number)
        groups.append(temp_buffer)
    return groups


def extract_data():
    file_name = r'data.txt'
    infile = open(file_name, 'r')
    temp_buffer = []
    for line in infile:
        temp_buffer.append(line.strip('\n'))
    temp_buffer = temp_buffer[8:]
    i = 0
    groups = []
    temp = []
    for item in temp_buffer:
        if i != 390:
            temp.append(item)
            i += 1
        else:
            groups.append(temp)
            temp = []
            i = 0
#             break
    groups.append(temp)
    infile.close()
    return groups


def main():
 
    #load data and plot figures, here data means the useful_data.
#    data=load_sample_data()
#    data_all = load_sample_data_all()

#    plot_data(data)
    #plot_data_all(data_all)

#    convert_image()
    # convert_image_all()

    #initial
#    p1 = get_pixel_values()
    p2 = get_pixel_values_all()
    
#    r1 = find_returns(data)
    
    r2 = find_returns(data_all)
#    np.savetxt("r1.txt", r1, fmt="%s", delimiter=",")
#    np.savetxt("r2.txt", r2, fmt="%s", delimiter=",")
#    print(y)
#    p = np.array((p1,p2))
#    r = np.array((r1,r2))   

    x = np.asarray(p2)
    y = np.asarray(r2)
    
    x_train = x[0:5000]
    y_train = y[0:5000]
    x_test = x[0:100]
    y_test = y[0:100]

    #y_true = y_test
    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test = np_utils.to_categorical(y_test, 2)
    x_train = x_train.astype('float32')
##    np.savetxt("x_train_float.txt", x_train, fmt="%s", delimiter=",") 
    x_test = x_test.astype('float32')
    #255 means white background, set all bg to '1'.
    x_train /= 255.0
    x_test /= 255.0
##    np.savetxt("x_train_255.txt", x_train, fmt="%s", delimiter=",") 

    #create model
    model = VGG_19(None)
    model = compile_model(model,lrate)
    print ("fit the model=============================")
    # Fit the model
    epochs = 100
##    model.fit(x_train, y_train, validation_data=(x_test, y_test), 
##              epochs=epochs,
##              shuffle=True, batch_size=100, verbose=1)
#   validation_split  the rate of the data used for verify
    model.fit(x_train, y_train, validation_split = 0.1, 
              epochs=epochs,
              shuffle=True, batch_size=100, verbose=2)
    #scores = model.evaluate(x_test, y_test, verbose=0)
    #print('Accuracy: %.2f%%' % (scores[1] * 100))
#    classes = model.predict_classes(x_test, verbose=0)
#    classes = list(classes)
#    y_test = list(y_test)
#    r2 = r_squared(y_test, classes)
#    print (r2)


if __name__ == '__main__':
    main()
