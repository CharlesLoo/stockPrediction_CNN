import matplotlib
import csv
import re
matplotlib.use('Agg')

from random import shuffle

import errno
import os
from glob import glob
import skimage.io
import skimage.transform
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_FOLDER = 'data'
HEIGHT = 224
WIDTH = 224

def read_tfrecord(folder):
    tfr_name = DATA_FOLDER+r'/tfrecord/'+folder+r'/img.tfrecors'
    filename_queue = tf.train.string_input_producer([tfr_name]) #creat a queue of files

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return filename and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })#get the image data and label
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [HEIGHT, WIDTH, 3])  #reshape
    img = tf.cast(img, tf.float32) * (1. / 255) # - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.int32) #在流中抛出label张量
    return img, label

def load_image(path):
    try:
        img = skimage.io.imread(path).astype(float)
        # TODO http://scikit-image.org/docs/dev/api/skimage.color.html rgb2gray
        # TODO cropping.
        img = skimage.transform.resize(img, (HEIGHT, WIDTH), mode='constant')
    except:
        return None
    if img is None:
        return None
    if len(img.shape) < 2:
        return None
    if len(img.shape) == 4:
        return None
    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] > 4:
        return None

    img /= 255.
    return img


def next_batch(x_y, index, batch_size):
    has_reset = False
    index *= batch_size
    #updated_index = index % len(x_y)
    beg = index
    end = beg + batch_size
    #end is out of length, need to reset
    if end > len(x_y):
        beg = 0
        end = beg + batch_size
        has_reset = True
    
    output = x_y[beg:end]
    x = np.array([e[0] for e in output])
    y = np.array([e[1] for e in output])
    return x, y, has_reset

#read image,(path,folder,number of image be read,)
def read_dataset(folder, max_num_of_train_images,max_num_of_test_images):
    
    #read label
    label_folder = DATA_FOLDER+r'/log_return/'+folder+r'/log_return.csv'
    csv_reader = csv.reader(open(label_folder, encoding='utf-8'))
    img_label = []
    for row in csv_reader:
        img_label.append(float(row[0]))

    #read img
    images_folder =  DATA_FOLDER+r'/figs/'+folder
    inputs = []
    #glob bug是乱序的，需要重新匹配顺序
    list_images = glob(images_folder + '/*.png')
    max_num_of_images = max_num_of_train_images + max_num_of_test_images
    shuffle(list_images)
    for i, image_name in enumerate(list_images):
        if len(inputs) >= max_num_of_images:
            break
        #获取图片序号
        base_name = os.path.basename(image_name)
        img_num = int(re.sub("\D","",base_name))
        #include a img  and a class "down or up" --> 0 or 1
        inputs.append([load_image(image_name), img_label[img_num]])  # TODO make them 256x256
    return inputs

#because to read all the images once need too much memory, here I only return a generator.
def read_data_by_folder(folder,max_num_of_train_images,max_num_of_test_images):
    inputdata = read_dataset(folder, max_num_of_train_images,max_num_of_test_images)
    print('read_dataset() in folder '+folder+' done')
    ##compute mean
    print('compute_mean() start')
    mean_image = compute_mean(inputdata)
    print('compute_mean() done')
    inputs = subtract_mean(inputdata, mean_image)
    print(len(inputs), 'inputs in folder '+folder)
        
    #divide the input into training sets and test sets
    inputs[:max_num_of_train_images]
    inputs[-max_num_of_test_images:]
    print(len(inputs[:max_num_of_train_images]), 'training inputs in folder '+folder)
    print(len(inputs[-max_num_of_test_images:]), 'testing inputs in folder '+folder)
    yield [inputs[:max_num_of_train_images],inputs[-max_num_of_test_images:]]

#to save memory     
def make_tfrecord(folder):
    #read label
    label_folder = DATA_FOLDER+r'/log_return/'+folder+r'/log_return.csv'
    csv_reader = csv.reader(open(label_folder, encoding='utf-8'))
    img_label = []
    img_name = []
    for row in csv_reader:
        img_label.append(float(row[0]))

    tfr_folder = DATA_FOLDER+r'/tfrecord/'+folder
    if os.path.exists(tfr_folder) == False:
       os.mkdir(tfr_folder)
    print(tfr_folder+r'/img.tfrecors')
    writer =tf.python_io.TFRecordWriter(tfr_folder+r'/img.tfrecors')
   
    images_folder =  DATA_FOLDER+r'/figs/'+folder
    list_images = glob(images_folder + '/*.png')#obtain all image path  
    for img_path in list_images: 
        img = Image.open(img_path)
        img = img.resize((HEIGHT,WIDTH))
        img_raw = img.tobytes() # change image to bytes 
        #获取图片序号
        base_name = os.path.basename(img_path)
        img_num = int(re.sub("\D","",base_name))
        img_lb = img_label[img_num] #image label
        img_tfr = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_lb)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #package label and image data
        writer.write(img_tfr.SerializeToString())
    writer.close()


def compute_mean_not_optimised(inputs):
    matrix_all_images = []
    for image, label in inputs:
        matrix_all_images.append(image)
    return np.mean(np.array(matrix_all_images), axis=0)


def compute_mean(inputs):
    image_mean = np.array(inputs[0][0])
    image_mean.fill(0)
    for i, (image, label) in enumerate(inputs):
        image_mean += image
        if i % 100 == 0:
            print(i)
    return image_mean / len(inputs)


def subtract_mean(inputs, mean_image):
    new_inputs = []
    for image, label in inputs:
        new_inputs.append([image - mean_image, label])
    return new_inputs

#mkdir
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def generate_time_series(arr, filename):
    generate_multi_time_series([arr], filename)


def generate_multi_time_series(arr_list, filename):
    fig = plt.figure()
    for arr in arr_list:
        plt.plot(arr)
    plt.savefig(filename)
    plt.close(fig)
