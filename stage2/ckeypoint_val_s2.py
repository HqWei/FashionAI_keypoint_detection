import os

import cv2
import numpy as np
import scipy.misc as misc
import tensorflow as tf


from stage2.config_s2 import Config

from net import fai_cpn_net as mnet

from utils import offline_val
from utils import util
from pandas import DataFrame
from tqdm import tqdm
columns  =[   'image_id','image_category',
              'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
              'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
              'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
              'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
              'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
              'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
]


config = Config()



os.environ['CUDA_VISIBLE_DEVICES'] = '3'

slim = tf.contrib.slim
category_classnum_dict = config.category_classnum_dict
all_labels = config.all_labels
category_label_dict = config.category_label_dict
img_category = config.img_category
category_change_index = config.category_change_index
img_size = config.IMAGE_SIZE
def demo(category,step):
    '''

    :param category: one of ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    :return:
    '''
    numclass = category_classnum_dict[category]
    category_labels = category_label_dict[category]


    #img_size_list = [int(384 * 0.5),int(384 * 1),int(384 * 1.5),int(384 * 2)]
    img_size_list = [int(img_size*1)]

    with tf.Graph().as_default():
        batch_x = tf.placeholder(shape=[1,None,None,3],dtype=tf.float32)

        with tf.variable_scope('cpn_model'):
            model1 = mnet.CPN(numclass,1)
            model1.build_model(batch_x,False)

        with tf.variable_scope('cdet'):
            model2 = mnet.CPN(numclass,1)
            model2.build_model(batch_x,False)


        with tf.Session() as sess:



            all_vars = slim.get_model_variables()

            vars1 = []
            vars2 = []
            for var in all_vars:
                if 'cpn_model' in var.op.name:
                    vars1.append(var)
                elif 'cdet' in var.op.name:
                    vars2.append(var)
                else:
                    raise ValueError('wrong init')

            ckpt_filename1 = '../stage1/trained_weights_s1/'+category
            checkpoint_path1 = tf.train.latest_checkpoint(ckpt_filename1)
            saver1 = tf.train.Saver(var_list=vars1)
            saver1.restore(sess, checkpoint_path1)

            ckpt_filename2 = 'trained_weights_s2/'+category
            checkpoint_path2 = ckpt_filename2+'/'+str(step)+'.ckpt'
            saver2 = tf.train.Saver(var_list=vars2)
            saver2.restore(sess, checkpoint_path2)

            dict_list = []
            f = open('../data/image_ori/val.txt')
            list_file = f.read().splitlines()
            for j in tqdm(range(len(list_file))):
                temp = list_file[j].split(',')
                category_t = temp[1]
                if category_t != category:
                    continue

                img_id = temp[0]

                img_full = misc.imread('../data/image_ori/'+img_id)
                img_full_ = img_full.copy()
                img_384_full, scale_384_full, start_index_384_full = util.make_for_input(img_full_, 384)

                img_384_full = cv2.cvtColor(img_384_full, cv2.COLOR_RGB2BGR) / 256.0 - 0.5
                img_384_full = np.expand_dims(img_384_full, 0)
                heat_for_box = sess.run(model1.finalout, feed_dict={batch_x: img_384_full})
                heat_for_box_m = heat_for_box[0,:,:,:]
                location_box = util.get_location_cpn_n(stage_heatmap=heat_for_box_m)

                location_box_ori = util.restore_location(ori_img_shape=img_full.shape, label_output=location_box, scale=scale_384_full,
                                                         start_index=start_index_384_full)
                label_box = np.array(location_box_ori)

                x = label_box[:, 1]
                y = label_box[:, 0]


                xd = 40
                yd = 30
                xmin = min(x)
                ymin = min(y)
                xmax = max(x)
                ymax = max(y)
                xmin = max(0, xmin - xd)
                xmax = min(img_full.shape[1], xmax + xd)
                ymin = max(0, ymin - yd)
                ymax = min(img_full.shape[0], ymax + yd)
                img = img_full[ymin:ymax, xmin:xmax, :]
                img_ = img.copy()



                _, scale_384, start_index_384 = util.make_for_input(img_, img_size)
                heat_scale = []
                for img_size_m in img_size_list:
                    img_scale, scale, start_index = util.make_for_input(img_, img_size_m)


                    img_input = cv2.cvtColor(img_scale, cv2.COLOR_RGB2BGR) / 256.0 - 0.5
                    img_input = np.expand_dims(img_input, 0)

                    img_2 = cv2.flip(img_, 1)

                    img_scale2, scale2, start_index2 = util.make_for_input(img_2, img_size_m)


                    img_input2 = cv2.cvtColor(img_scale2, cv2.COLOR_RGB2BGR) / 256.0 - 0.5
                    img_input2 = np.expand_dims(img_input2, 0)

                    stage_heatmap_n = sess.run(model2.finalout, feed_dict={batch_x: img_input})

                    stage_heatmap_n2 = sess.run(model2.finalout, feed_dict={batch_x: img_input2})

                    t1 = stage_heatmap_n[0,:,:,:]
                    t2 = stage_heatmap_n2[0, :, :, :]
                    t2 = cv2.flip(t2,1)


                    left_index = category_change_index[category][0]
                    right_index = category_change_index[category][1]


                    for z in range(len(left_index)):
                        temp = np.copy(t2[:, :, left_index[z]])
                        t2[:, :, left_index[z]] = np.copy(t2[:, :, right_index[z]])
                        t2[:, :, right_index[z]] = np.copy(temp)

                    tt = (t1 + t2) / 2.0

                    tt_384 = cv2.resize(tt, (img_size//4, img_size//4))
                    heat_scale.append(tt_384)
                heat_scale = np.array(heat_scale).transpose(1,2,3,0)

                heat_scale_m = np.mean(heat_scale,axis=-1)


                location_output = util.get_location_cpn_n(stage_heatmap=heat_scale_m)  # [y,x]

                location_in_ori = util.restore_location(ori_img_shape=img.shape, label_output=location_output, scale=scale_384,
                                                         start_index=start_index_384)

                location_in_ori = np.array(location_in_ori)
                location_in_full = np.copy(location_in_ori)
                for tt in range(location_in_ori.shape[0]):
                    location_in_full[tt, 0] = min(img_full.shape[0], location_in_ori[tt, 0] + ymin)
                    location_in_full[tt, 1] = min(img_full.shape[1], location_in_ori[tt, 1] + xmin)

                dict_t = {}
                dict_t['image_id'] = img_id
                dict_t['image_category'] = category
                i = 0
                for label in all_labels:
                    if label in category_labels:
                        dict_t[label] = str(location_in_full[i][1]) + '_' + str(location_in_full[i][0]) + '_' + str(1)
                        i += 1
                    else:
                        dict_t[label] = '-1_-1_-1'
                dict_list.append(dict_t)

            test_data = DataFrame(data=dict_list, columns=columns)
            f.close()
    return test_data



# ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
# img_clist = [['blouse',[30000]],
#              ['skirt',[39000]],
#              ['outwear', [60000]],
#              ['dress', [30000]],
#              ['trousers', [50000]]]


img_clist = [['blouse',[30000]]]
for clist in img_clist:
    category = clist[0]
    steps=clist[1]



    f = open('acc_log/' + category + '.txt', 'a')
    for step in steps:
        test_data = demo(category,step)

        test_data.to_csv('../temp_result/'+category+'_'+str(step)+'.csv', index=False, sep=',')

        score = offline_val.offval(category,step)

        f.write('p'+'\t'+str(step)+'\t'+str(score)+'\n')
        print(category)
        print (step,score)
    f.close()
