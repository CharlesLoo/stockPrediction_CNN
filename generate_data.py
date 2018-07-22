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
import os  
from alexnet_data import make_tfrecord
#seed = 7
#np.random.seed(seed)
width = 5
height = 5

def extract_useful_data_all(data):
    groups = []
    for group in data:
        temp_buffer = []
        for item in group:
            temp = [item[0],item[1],item[2], item[3]]
            temp = [float(i) for i in temp]
            temp_buffer.append(temp)
        groups.append(temp_buffer)
    return groups


def split_data_all(data):
    groups = []
    for item in data:
        temp_buffer = []
        for string in item:
            number = string.split(',')
            temp_buffer.append(number)
        groups.append(temp_buffer)
    return groups


def extract_data_all(data_path):
    groups = []
    files= os.listdir(data_path) #得到文件夹下的所有文件名称  
    for file1 in files:
      file_name = r''+data_path+'/'+file1
      print(r'processing '+file_name+' ...')
      infile = open(file_name, 'r')
      temp_buffer = []
      for line in infile:
          if line.find('nan') == -1: #remove all nan values
             temp_buffer.append(line.strip('\n'))
      ##一个文件一个group
      groups.append(temp_buffer)
      infile.close()
    return groups

def load_sample_data_all(data_path):
    original_data_all = extract_data_all(data_path)
##    np.savetxt("original_data_all.txt", original_data_all, fmt="%s", delimiter=",") 
    splitted_data_all = split_data_all(original_data_all)
##    np.savetxt("splitted_data_all.txt", splitted_data_all, fmt="%s", delimiter=",") 
    useful_data_all = extract_useful_data_all(splitted_data_all)
##    np.savetxt("useful_data_all.txt", useful_data_all, fmt="%s", delimiter=",") 
    return useful_data_all

#using price in previous 100 days to predict price of 10 days late.
def plot_data_all(data,fig_path):
    t = np.arange(0, 59, 1)
    file_name_number = 0
    fig = plt.figure(frameon=False, figsize=(width, height))
    for group in data:
        count = 60
        while count <= (len(group)-10):
            close = []
            high = []
            low = []
            op = []
##            vol = []
            for item in group[count-60:count]:
                op.append(item[0])
                high.append(item[1])
                low.append(item[2])
                close.append(item[3])
                #vol is much bigger, need to make it smaller to put them in the same img
#                vol.append(item[2]/1000000)
            file_name = r'fig_' + str(file_name_number)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(t, close[0:-1], 'y',t, high[0:-1], 'b', t, low[0:-1], 'g',t, op[0:-1], 'r')
            if os.path.exists(fig_path) == False:
               os.mkdir(fig_path)
            fig.savefig(fig_path+'/' + file_name, dpi=100)
            fig.clf()
            file_name_number += 1
            count += 60
    print('Created %d files!' % file_name_number)

def convert_image_all(fig_path):
    file_name = fig_path
    for filename in glob.glob(file_name + '/*.png'):
        img = Image.open(filename)
        img = img.convert('RGB')
        img.save(filename)

#compute the logarithmic return (like label) 
# 100d  --> predict 10d late  
def find_returns_all(data,r_path):
    returns = []
    log_return = []
    for group in data:
        count = 60
        index = 1
        while count <= (len(group)-10):
            current_data = group[count-1]
            future_data = group[count+9]
            #coz high and low normally increas and decreas together, we use their mean to present the whole value
            p1 = np.mean(current_data)
            p2 = np.mean(future_data)
            #logarithmic return
            log_ret = math.log(p2/p1)
#            log_return.append(log_ret)
            if log_ret >= 0:
               returns.append(1)
            else:
               returns.append(0)
            count += 60
            index += 1
    if os.path.exists(r_path) == False:
       os.mkdir(r_path)
    np.savetxt(r_path+r'/log_return.csv', returns, delimiter=",") 
    return returns
    
def get_pixel_values_all(fig_path):
    file_name = fig_path
    pixels = []
    for filename in glob.glob(file_name + '/*.png'):
        print(filename)
        im = misc.imread(filename)
        pixels.append(im)
    return pixels


def main_process_data(path):
    csv_path = path+r'/csv_data'
    csv_folders= os.listdir(csv_path) #get all csv folders in the path
    for folder in csv_folders:
       csv_folder = csv_path+r'/'+folder
       fig_folder = path+r'/figs/'+folder
       log_return_folder = path+r'/log_return/'+folder
       
    #   load data and plot figures, here data means the useful_data.
       #data = load_sample_data_all(csv_folder)
       #plot_data_all(data,fig_folder)
       #convert_image_all(fig_folder)
       #find_returns_all(data,log_return_folder)

    #make tfrecord
    folder_list = os.listdir(os.path.join(path,'figs'))
    for folder in folder_list:
        make_tfrecord(folder)
    print("make tfrecord successful")

if __name__ == '__main__':
    path = 'data'
    main_process_data(path)
