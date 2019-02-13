import math
import os

import cv2
import numpy as np
import scipy.misc as misc
import tensorflow as tf
from stage2.config_s2 import Config
config = Config()

# get config from unity config class
img_category = config.img_category

blouse_labels = config.blouse_labels

skirt_labels = config.skirt_labels

outwear_labels =config.outwear_labels

dress_labels = config.dress_labels

trousers_labels = config.trousers_labels

all_labels = config.all_labels

category_label_dict = config.category_label_dict
category_classnum_dict = config.category_classnum_dict

img_size = config.IMAGE_SIZE

def make_tfrecord(category_set,anofile_path,tffile_path,img_path):
    '''

    :param category_set: one of ['blouse', 'skirt', 'outwear', 'dress', 'trousers']
    :param anofile_path: annotation file path
    :param tffile_path:  the tfrecord file save path
    :param img_path:     img data path
    :return:
    '''

    if not os.path.exists(tffile_path):
        os.mkdir(tffile_path)

    if not os.path.exists(anofile_path):
        raise ValueError("wrong anofile path")
    if not os.path.exists(img_path):
        raise ValueError("wrong img path")


    filename = tffile_path+category_set+'.tfrecord'
    writer = tf.python_io.TFRecordWriter(filename)
    f= open(anofile_path)
    ano_file = f.read().splitlines()
    for x in range(len(ano_file)):
        print (x)
        temp = ano_file[x].split(',')
        category = temp[1]

        if category!=category_set:
            continue

        img_id = temp[0]
        img = misc.imread(img_path + img_id)




        label_index = [all_labels.index(t)+2 for t in category_label_dict[category]]



        labelt2 = []
        for ii in label_index:
            ti = temp[ii].split('_')
            if ti[-1]=='-1':
                labelt2.append(['0','0'])
            else:
                labelt2.append(ti[:2])


        labelt2 = [  [z.replace('-1','0') for z in x ] for x in labelt2]
        labelt2 = np.array(labelt2,dtype=np.int32)
        def squ_zero(label_in):
            label_out = np.copy(label_in)
            for l in range(label_in.shape[0]):
                if label_in[l,0]==0 and label_in[l,1]==0:
                    label_out[l,0]=999
                    label_out[l,1]=999
            return label_out
        label2 = squ_zero(labelt2)

        x = label2[:,0]
        y = label2[:,1]

        xd=40
        yd=30
        xmin = min(x)
        ymin = min(y)
        x[x==999]=0
        y[y==999]=0
        xmax = max(x)
        ymax = max(y)

        xmin = max(0,xmin-xd)
        xmax = min(img.shape[1],xmax+xd)
        ymin = max(0,ymin-yd)
        ymax = min(img.shape[0],ymax+yd)
        img_cut = img[ymin:ymax,xmin:xmax,:]



        labelt = []
        for ii in label_index:
            ti = temp[ii].split('_')
            if ti[-1]=='-1' or ti[-1]=='0':
                labelt.append(['0','0'])
            else:
                labelt.append(ti[:2])


        labelt = [  [z.replace('-1','0') for z in x ] for x in labelt]
        labelt = np.array(labelt,dtype=np.int32)


        label_cut = np.copy(labelt)
        for tt in range(labelt.shape[0]):
            label_cut[tt,0] = max(0,labelt[tt,0]-xmin)
            label_cut[tt,1] = max(0,labelt[tt,1]-ymin)



        # the value save to tfrecord

        img_scale,label_scale = _resize_img_label(img_cut,img_size,label_cut)

        im_raw = img_scale.tobytes()
        label_raw = label_scale.tobytes()
        #view_label_raw = view_label.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_id.encode()])), # seems no use
            'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw])),
            'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))
        }))

        writer.write(example.SerializeToString())
    print ('make '+category_set+' done')
    writer.close()
    f.close()



def make_gaussian(x_size,y_size, fwhm=11, center=None):
    '''
    make gaussion
    :param x_size: width of created map
    :param y_size: height of created map
    :param fwhm:   size of gaussian kernel
    :param center: [x,y] label location
    :return: the created map
    '''
    x = np.arange(0, x_size, 1, float)
    y_t = np.arange(0,y_size,1,float)
    y = y_t[:, np.newaxis]

    x0 = center[0]
    y0 = center[1]

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)


def _resize_img_label(img,re_size,label_ori):
    '''
    scale the img according to re_size and put it in the center, and scale the label

    :param img: the img you want to resize
    :param re_size: the size you want to get
    :param label_ori: the label before scale
    :return: resized img and label
    '''

    img_scale = np.ones([re_size,re_size,3],dtype=np.uint8)*128
    height = img.shape[0]
    width = img.shape[1]
    scale = float(re_size)/max(height,width)
    img_ = img.copy()
    img2=cv2.resize(img_,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    label_scale = [[int(math.ceil(sl[0]*scale)),int(math.ceil(sl[1]*scale))] for sl in label_ori]
    height_n = img2.shape[0]
    width_n = img2.shape[1]

    start_index = int(re_size/2-math.ceil(min(height_n,width_n)/2))

    def uf(s,xstart):
        if s[0]==0 and s[1]==0:
            return [0,0]
        else:
            return [s[0], s[1]+xstart]

    def uf2(s,xstart):
        if s[0]==0 and s[1]==0:
            return [0,0]
        else:
            return [s[0] + xstart, s[1]]

    if height_n<=width_n:
        img_scale[start_index:start_index+height_n,:,:]=img2
        label_scale = [uf(sl,start_index) for sl in label_scale]
    else:
        img_scale[:,start_index:start_index+width_n,:]=img2
        label_scale = [uf2(sl,start_index) for sl in label_scale]

    return img_scale,np.array(label_scale,dtype=np.int32)



if __name__ == '__main__':

    # make val data or train data
    sub = ['blouse']
    # img_category

    img_path = '../data/image_ori/'

    # the paht to save tfreocord
    tffile_path_train = '../data/tfrecord_s2/train/'
    anofile_path_train = '../data/image_ori/train.txt'
    for category in sub:
        make_tfrecord(category, anofile_path_train, tffile_path_train, img_path)

    # the paht to save tfreocord
    tffile_path_val = '../data/tfrecord_s2/val/'
    anofile_path_val = '../data/image_ori/val.txt'
    for category in sub:
        make_tfrecord(category, anofile_path_val, tffile_path_val, img_path)




