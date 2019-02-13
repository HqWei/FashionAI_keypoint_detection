import cv2
import numpy as np
import scipy.misc as misc
import tensorflow as tf
from pandas import DataFrame
from tqdm import tqdm

from stage1.config_s1 import Config
from net import fai_cpn_net as mnet
import os
from utils import offline_val
from utils import util
config = Config()
slim = tf.contrib.slim
#param

columns  =[   'image_id','image_category',
              'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
              'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
              'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
              'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
              'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
              'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
]

category_classnum_dict = config.category_classnum_dict
all_labels = config.all_labels
category_label_dict = config.category_label_dict
img_category = config.img_category
category_change_index = config.category_change_index
img_size = config.IMAGE_SIZE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def demo(category,step):
    '''
    :param category: one of ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    :return:
    '''
    numclass = category_classnum_dict[category]
    category_labels = category_label_dict[category]
    img_size_list = [int(img_size * 0.5),int(img_size * 1),int(img_size * 1.5),]

    with tf.Graph().as_default():
        batch_x = tf.placeholder(shape=[1,None,None,3],dtype=tf.float32)
        with tf.variable_scope('cpn_model'):
            model = mnet.CPN(numclass, 1)
            model.build_model(batch_x,False)


        with tf.Session() as sess:


            saver = tf.train.Saver()

            ckpt_filename = 'trained_weights_s1/'+category
            checkpoint_path = ckpt_filename+'/'+str(step)+'.ckpt'
            saver.restore(sess, checkpoint_path)

            dict_list = []
            f = open('../data/image_ori/val.txt')
            list_file = f.read().splitlines()
            for x in tqdm(range(len(list_file))):
                temp = list_file[x].split(',')
                category_t = temp[1]
                if category_t != category:
                    continue

                img_id = temp[0]
                img = misc.imread('../data/image_ori/'+img_id)
                img_ = img.copy()

                _, scale_384, start_index_384 = util.make_for_input(img_, 512)
                heat_scale = []
                for img_size_m in img_size_list:
                    img_scale, scale, start_index = util.make_for_input(img_, img_size_m)

                    img_input = cv2.cvtColor(img_scale, cv2.COLOR_RGB2BGR) / 256.0 - 0.5
                    img_input = np.expand_dims(img_input, 0)

                    img_2 = cv2.flip(img_, 1)

                    img_scale2, scale2, start_index2 = util.make_for_input(img_2, img_size_m)

                    img_input2 = cv2.cvtColor(img_scale2, cv2.COLOR_RGB2BGR) / 256.0 - 0.5
                    img_input2 = np.expand_dims(img_input2, 0)

                    stage_heatmap_n = sess.run(model.finalout, feed_dict={batch_x: img_input})

                    stage_heatmap_n2 = sess.run(model.finalout, feed_dict={batch_x: img_input2})

                    t1 = stage_heatmap_n[0, :, :, :]
                    t2 = stage_heatmap_n2[0, :, :, :]
                    t2 = cv2.flip(t2, 1)

                    left_index = category_change_index[category][0]
                    right_index = category_change_index[category][1]

                    for z in range(len(left_index)):
                        temp = np.copy(t2[:, :, left_index[z]])
                        t2[:, :, left_index[z]] = np.copy(t2[:, :, right_index[z]])
                        t2[:, :, right_index[z]] = np.copy(temp)

                    tt = (t1 + t2) / 2.0

                    tt_384 = cv2.resize(tt, (img_size//4, img_size//4))
                    heat_scale.append(tt_384)
                heat_scale = np.array(heat_scale).transpose(1, 2, 3, 0)

                heat_scale_m = np.mean(heat_scale, axis=-1)


                location_output = util.get_location_cpn_n(stage_heatmap=heat_scale_m)  # [y,x]

                location_in_ori = util.restore_location(ori_img_shape=img.shape, label_output=location_output,
                                                         scale=scale_384,
                                                         start_index=start_index_384)



                dict_t = {}
                dict_t['image_id'] = img_id
                dict_t['image_category'] = category
                i = 0
                for label in all_labels:
                    if label in category_labels:
                        dict_t[label] = str(location_in_ori[i][1]) + '_' + str(location_in_ori[i][0]) + '_' + str(1)
                        i += 1
                    else:
                        dict_t[label] = '-1_-1_-1'
                dict_list.append(dict_t)

            test_data = DataFrame(data=dict_list, columns=columns)
            f.close()
    return test_data


#['blouse', 'skirt', 'outwear', 'dress', 'trousers']
cate_set = ['blouse']
for category in cate_set:
    f = open('acc_log/' + category + '.txt','a')
    for step in [46000]:

        test_data = demo(category,step)

        test_data.to_csv('../temp_result/'+category+'_'+str(step)+'.csv', index=False, sep=',')

        score = offline_val.offval(category,step)

        f.write('p'+'\t'+str(step)+'\t'+str(score)+'\n')
        print(category)
        print (step,score)
    f.close()

