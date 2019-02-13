import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from input_lib import data_input_mask as data_input
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'



is_inspect_s1=False
if is_inspect_s1:
    from stage1.config_s1 import Config
    data_path_pro = '../data/tfrecord_s1/val/'
else:
    from stage2.config_s2 import Config
    data_path_pro = '../data/tfrecord_s2/val/'


config = Config()
category_change_index = config.category_change_index
category_classnum_dict = config.category_classnum_dict
data_input.img_size=config.IMAGE_SIZE


category = 'blouse'

data_path = data_path_pro+category+'.tfrecord'
(batch_x,batch_y,batch_point_mask)= data_input.read_batch(data_path, category_classnum_dict[category],category_change_index[category],False,config.IMAGE_SIZE,16)
gt_heatmap_cal = tf.image.resize_images(batch_y, (config.IMAGE_SIZE,config.IMAGE_SIZE))
with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    for i in range(100):

        x_n,y_n,gt,pm = sess.run([batch_x,batch_y,gt_heatmap_cal,batch_point_mask])

        print(pm[0])

        for x in range(category_classnum_dict[category]):

            plt.imshow(np.uint8((x_n[0]+0.5)*256))
            plt.imshow(gt[0, :, :, x],alpha=0.5)
            plt.show()



    coord.request_stop()
    coord.join(threads)


