import os

import numpy as np
import tensorflow as tf
from PIL import Image

from alexnet_data import read_dataset, next_batch, compute_mean, subtract_mean,DATA_FOLDER,read_data_by_folder,make_tfrecord,read_tfrecord

if __name__ == '__main__':
    
    Toall_num_of_train_images = 1917 #the number of pictures in the train folder
    Toall_num_of_test_images = 581 #the number of pictures in the test folder
    
    max_num_of_train_images = 2000 # the maxnumber in the quenu
    max_num_of_test_images = 400 # the maxnumber in the quenu

    epochs = int(1e9)

    #names = os.listdir(os.path.join(DATA_FOLDER, 'train'))
    
    BATCH_SIZE = 1
    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3
    LEARNING_RATE = 0.1

    
    ##read dataset
    print('read_dataset() start')
    folder_list = os.listdir(os.path.join(DATA_FOLDER,'figs'))
    num_folder = len(folder_list)
    dataset_list = []

    #read tfrecord
    print("reading tfrecord file")
    tr_img,tr_label = read_tfrecord('2')
    te_img,te_label = read_tfrecord('2')

    #initialize network
    x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNELS])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)

    from alexnet_keras import alex_net_keras
    logits = alex_net_keras(x, num_classes=2, keep_prob=keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #config tensorflow
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 占用GPU80%的显存
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())   

    img_batch, label_batch = tf.train.batch([tr_img, tr_label],
                                                batch_size=BATCH_SIZE, capacity=max_num_of_train_images)

    te_img_batch, te_label_batch = tf.train.shuffle_batch([te_img, te_label],
                                                batch_size=BATCH_SIZE, capacity=max_num_of_test_images,
                                                min_after_dequeue=int(max_num_of_test_images/5))
    total_train_batches = 0
    #innitial the results file
    te_result_file = r'results/te_loss_acc_vgg_180_30_60_BS'+str(BATCH_SIZE)+r'_LR'+str(LEARNING_RATE)+r'_Trn'+str(Toall_num_of_train_images)+r'_Ten'+str(Toall_num_of_test_images)+r'.csv'
    te_output = open(te_result_file,'w')
    string = 'total_train_batches,loss,accuracy\n'
    te_output.write(string)
    te_output.close()

    tr_result_file = r'results/tr_loss_acc_vgg_180_30_60_BS'+str(BATCH_SIZE)+r'_LR'+str(LEARNING_RATE)+r'_Trn'+str(Toall_num_of_train_images)+r'_Ten'+str(Toall_num_of_test_images)+r'.csv'
    tr_output = open(tr_result_file,'w')
    string = 'total_train_batches,loss,accuracy\n'
    tr_output.write(string)
    tr_output.close()

    for epoch in range(epochs): 
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        #computing the mean
        for i in range(1): 
           image,label=sess.run([img_batch,label_batch])
           image_mean = np.array(image[0])
           image_mean.fill(0)
        
        for i in range(int(Toall_num_of_train_images/BATCH_SIZE)+1):
                image,label=sess.run([img_batch,label_batch])

                for j,img in enumerate(image):
                    image_mean += img
                    if j == BATCH_SIZE:
                       print(j)
        image_mean  = image_mean/((int(Toall_num_of_train_images/BATCH_SIZE)+1)*BATCH_SIZE)
        print(image_mean)

        while not coord.should_stop():
            #train     
            print("traing ============================================================================")
            #for i in range(int(max_num_of_train_images/BATCH_SIZE)):
            for i in range(20):
                image,label=sess.run([img_batch,label_batch])
                imgs = []
                #substract_mean
                for img in image:
                    img = img - image_mean
                    imgs.append(img)
                image = []
                tr_loss, _ = sess.run([cross_entropy, train_step],
                            feed_dict={x: imgs, y: label, keep_prob: 0.5})
                total_train_batches += 1
                print(total_train_batches)

            #train_accuracy
            tr_accuracy_list = []
            tr_loss_list = [] 
            #for i in range(int(max_num_of_test_images/BATCH_SIZE)):
            for i in range(2):
                image, label= sess.run([img_batch, label_batch])
                imgs = []
                #substract_mean
                for img in image:
                    img = img - image_mean
                    imgs.append(img)
                image = []
                tr_loss, tr_acc = sess.run([cross_entropy, accuracy], feed_dict={x: imgs, y: label, keep_prob: 1.0})
                tr_accuracy_list.append(tr_acc)
                tr_loss_list.append(tr_loss)
            #print('[TRAINING] # epoch= {0}, tr_loss = {1:.3f}, training mean accuracy on training set = {2:.3f}'.format(epoch, tr_loss,   np.mean(tr_accuracy_list)))

            #test
            te_accuracy_list = []
            te_loss_list = [] 
            print("testing   =======================================================================")
#            for i in range(int(max_num_of_test_images/BATCH_SIZE)):
            for i in range(2):
                image, label= sess.run([te_img_batch, te_label_batch])
                imgs = []
                #substract_mean
                for img in image:
                    img = img - image_mean
                    imgs.append(img)
                image = []
                te_loss, te_acc = sess.run([cross_entropy, accuracy],
                                               feed_dict={x: imgs, y: label, keep_prob: 1.0})
                te_accuracy_list.append(te_acc)
                te_loss_list.append(te_loss)
            #print('[TESTING] total_train_batches= {0},  testing mean accuracy on testing set = {1:.2f}'.format(total_train_batches, np.mean(te_accuracy_list)))

            #save train reaults 
            tr_output = open(tr_result_file,'a')
            string = str(total_train_batches)+','+str(np.mean(tr_loss_list))+','+str(np.mean(tr_accuracy_list))+'\n'
            tr_output.write(string)
            tr_output.close() 
 
            #save test reaults           
            te_output = open(te_result_file,'a')
            string = str(total_train_batches)+','+str(np.mean(te_loss_list))+','+str(np.mean(te_accuracy_list))+'\n'
            te_output.write(string)
            te_output.close()
            
             #save model
            if (total_train_batches % 1000 == 0 and total_train_batches > 0):
                print("Saving check point")
                saver = tf.train.Saver()
                save_path = saver.save(sess,"check_point/save_net.ckpt")
                print("Saved check point") 

        coord.join(threads)    
    sess.close()
         














    

























