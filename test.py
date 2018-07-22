import os

import numpy as np
import tensorflow as tf
from PIL import Image
from alexnet_data import read_dataset, next_batch, compute_mean, subtract_mean,DATA_FOLDER,read_data_by_folder,make_tfrecord,read_tfrecord
folder_list = os.listdir(os.path.join(DATA_FOLDER,'figs'))

#for folder in folder_list:
#    print("ss")
#    make_tfrecord(folder)

img,label = read_tfrecord('2')

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.batch([img, label],
                                                batch_size=30, capacity=2000)

te_img_batch, te_label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=200,
                                                min_after_dequeue=50)

#img_batch, label_batch = tf.train.batch([img, label],
#                                                batch_size=30,capacity=1)
#print(img_batch)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    #image,label=sess.run([img_batch,label_batch])
    #print(label)
    
    for batch_num in range(int(2000/100)):     
      #for i in range(90):
      #  image, l= sess.run([img_batch, label_batch])
      #  for j in range(30):
          #我们也可以根据需要对val， l进行处理
      #    image[j]=(image[j]+0.5)*255
      #    ar=np.asarray(image[j],np.uint8)
      #    img=Image.frombytes("P",(224,224),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
      #    img.save("test_img2/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看

#      for i in range(10):
#        image, l= sess.run([img_batch, label_batch])
#        for j in range(30):
          #我们也可以根据需要对val， l进行处理
#          image[j]=(image[j]+0.5)*255
#          ar=np.asarray(image[j],np.uint8)
#          img=Image.frombytes("P",(224,224),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
#          img.save("test_img2/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看

      for i in range(10):
        image, l= sess.run([te_img_batch, te_label_batch])
        for j in range(30):
          #我们也可以根据需要对val， l进行处理
          image[j]=(image[j]+0.5)*255
          ar=np.asarray(image[j],np.uint8)
          img=Image.frombytes("P",(224,224),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
          img.save("test_img3/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看

      for i in range(10):
        image, l= sess.run([te_img_batch, te_label_batch])
        for j in range(30):
          #我们也可以根据需要对val， l进行处理
          image[j]=(image[j]+0.5)*255
          ar=np.asarray(image[j],np.uint8)
          img=Image.frombytes("P",(224,224),ar.tostring())#函数参数中宽度高度要注意。构建24×16的图片
          img.save("test_img2/reverse_%d.bmp"%(i+j),"BMP")#保存部分图片查看
      break


