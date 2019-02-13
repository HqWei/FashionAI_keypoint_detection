import tensorflow as tf
import numpy as np


def make_gaussian_tf(img_shape,fwhm=7,center=None):
    '''

    :param img_shape: the shape of created gaussian heatmaps
    :param fwhm:  the size of gassian kernel
    :param center: [x,y] location of keypoint
    :return: the created gaussian heatmaps
    '''

    x_tensor = tf.range(0, img_shape[1], 1,tf.float32)
    y_tensor = tf.expand_dims(tf.range(0, img_shape[0], 1, tf.float32),-1)


    g_map = tf.exp(-((x_tensor - center[0]) ** 2 + (y_tensor - center[1]) ** 2) / 2.0 / fwhm / fwhm)

    back_tensor = tf.zeros([img_shape[0], img_shape[1]], tf.float32)
    x0y0 = tf.add(center[0], center[1])
    map = tf.cond(tf.less(x0y0, 1), lambda: back_tensor, lambda: g_map)
    return map

def read_and_decode(tfr_queue,numclass,change_index,img_size,argument):
    '''

    :param tfr_queue: tfrecord file path(or a sequence )
    :param numclass: the numclass of corresponding category, an int value
    :param argument: do data argument?  Ture or False
    :return: the decode data
    '''
    tfr_reader = tf.TFRecordReader()
    _, serialized_example = tfr_reader.read(tfr_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_ids':tf.FixedLenFeature([],tf.string),
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.FixedLenFeature([], tf.string),
                                       })

    # decode data
    img_shape = [img_size,img_size,3]
    img_id = features['img_ids']
    img = tf.decode_raw(features['images'], tf.uint8)
    img = tf.reshape(img, img_shape)
    img = tf.cast(img, tf.float32)

    label_loc = tf.decode_raw(features['labels'], tf.int32)
    label_loc = tf.reshape(label_loc, [numclass, 2])
    label_loc = tf.cast(label_loc, tf.float32)
    do_flip = False
    if argument:
        if np.random.uniform(0.,1.)>0.5:
            do_flip = True
            left_index = change_index[0]
            right_index = change_index[1]

            label_loc = tf.split(label_loc, numclass)
            left_part = [label_loc[x] for x in left_index]
            right_part = [label_loc[x] for x in right_index]
            for x in range(len(left_index)):
                label_loc[left_index[x]] = right_part[x]
                label_loc[right_index[x]] = left_part[x]
            label_loc = tf.concat(label_loc, axis=0)

        else:
            do_flip = False

    heat_map_set = []
    for t in range(numclass):
        heat_map = make_gaussian_tf(img_shape=img_shape,fwhm=28,center=label_loc[t])*255.
        heat_map_set.append(heat_map)


    heat_map_set = tf.reshape(heat_map_set, [numclass, img_shape[0], img_shape[1]])
    heat_map_set = tf.transpose(heat_map_set, [1, 2, 0])
    heatmap = tf.cast(heat_map_set, tf.float32)

    if argument:
        max_hue_delta = 0.15
        low_sat = 0.7
        high_sat = 1.3
        max_bright_delta = 0.3


        # random hue
        img = tf.image.random_hue(img, max_delta=max_hue_delta)
        #
        #
        # # random saturation
        img = tf.image.random_saturation(img, lower=low_sat, upper=high_sat)
        #
        #
        # random brightness
        img = tf.image.random_brightness(img, max_delta=max_bright_delta)

    r,g,b = tf.split(img,3,2)
    img_bgr = tf.concat([b,g,r],2)


    if argument:




        #merge img + centermap + heatmap
        merged_img_heatmap = tf.concat([img_bgr, heatmap], axis=2)

        # subtract mean before pad
        mean_volume = tf.concat([128 * tf.ones(shape=(img_size, img_size, 3)),
                                 tf.zeros(shape=(img_size, img_size, numclass))], axis=2)

        merged_img_heatmap -= mean_volume

        merged_img_heatmap = preprocess( merged_img_heatmap,
                                         img_size,
                                         crop_off_ratio_y=0.0,
                                         crop_off_ratio_x=0.0,
                                         rotation_angle=0.3,
                                         scale_range=(0.7,1.3),  #0.7 1.3
                                         do_flip = do_flip
                                         )


        merged_img_heatmap += tf.concat((128 * tf.ones(shape=(img_size, img_size, 3)),
                                                        tf.zeros(shape=(img_size, img_size, numclass))), axis=2)

        preprocessed_img, preprocessed_heatmaps = tf.split(merged_img_heatmap, [3,numclass], axis=2)

        # preprocessed_img = tf.image.resize_images(preprocessed_img,(384,384))
        preprocessed_heatmaps = tf.image.resize_images(preprocessed_heatmaps,(int(img_size/4),int(img_size/4))) #(96,96,num)
        point_mask = tf.reduce_sum(preprocessed_heatmaps,[0,1])
        point_mask = tf.less(tf.constant(10000.0),point_mask)
        point_mask = tf.cast(point_mask,tf.float32)
        # Normalize image value
        preprocessed_img = preprocessed_img/256 -0.5

    else:
        #preprocessed_img = tf.image.resize_images(img_bgr, (384, 384))
        preprocessed_img = img_bgr/256 -0.5
        preprocessed_heatmaps = tf.image.resize_images(heatmap,(int(img_size/4),int(img_size/4)))

        point_mask = tf.reduce_sum(preprocessed_heatmaps,[0,1])
        point_mask = tf.less(tf.constant(10000.0),point_mask)
        point_mask = tf.cast(point_mask,tf.float32)



    return preprocessed_img, preprocessed_heatmaps,point_mask

def read_batch(tfr_path,numclass,change_index,argument,img_size,batch_size=1):
    '''

    :param tfr_path: tfrecord file path
    :param numclass: the numclass of corresponding category, an int value
    :param batch_size: bs
    :return:  the decode data
    '''


    with tf.name_scope('Inputs'):

        filename_queue = tf.train.string_input_producer([tfr_path],shuffle=True,capacity=100) # keep this function for multi-file train later

        (batch_images, batch_labels,batch_ids) = read_and_decode(filename_queue,numclass,change_index,img_size,argument=argument)

        #  something need to figure out
        # when I set a high value for capacity, the train process will stop when the read_and_decode process can't feed engouth data
        #  to shuffle_batch, so I set a low value (3batch) for it , but a low value for capacity can't get enough shuffle sequence
        (batch_images, batch_labels,batch_ids) = tf.train.shuffle_batch([batch_images, batch_labels,batch_ids],
                                                               batch_size=batch_size,
                                                               capacity=6 * batch_size,
                                                               min_after_dequeue=3*batch_size)

    return (batch_images, batch_labels,batch_ids)


def read_batch_val(tfr_path,numclass,change_index,argument,img_size,batch_size=1):
    '''

    :param tfr_path: val tfrecord file path
    :param numclass:
    :param batch_size:
    :return:
    '''


    with tf.name_scope('Inputs'):

        filename_queue = tf.train.string_input_producer([tfr_path],shuffle=False,capacity=100) # keep this function for multi-file train later

        (batch_images, batch_labels,batch_ids) = read_and_decode(filename_queue,numclass,change_index,img_size,argument=argument)


        (batch_images, batch_labels,batch_ids) = tf.train.batch([batch_images, batch_labels,batch_ids],
                                                               batch_size=batch_size,
                                                               capacity=3 * batch_size)

    return (batch_images, batch_labels,batch_ids)




def preprocess(image,
               img_size,
               rotation_angle=1.0,
               crop_off_ratio_y=0.1,
               crop_off_ratio_x=0.2,
               scale_range=(0.8,1.2),
               do_flip = False):
    """

    :param image: image to argument
    :param rotation_angle:
    :param crop_off_ratio: maximum cropping offset of top-left corner
    :return:
    """

    # [height, width, channel] of input image
    #img_shape_list = image.get_shape().as_list()
    img_shape_list = [img_size,img_size]



    # crop image
    new_top_left_x = crop_off_ratio_x * tf.random_uniform([], minval=0, maxval=1.0)

    new_top_left_y = crop_off_ratio_y * tf.random_uniform([], minval=0, maxval=1.0)


    new_bot_right_x = crop_off_ratio_x * tf.random_uniform([], minval=0, maxval=1.0)


    new_bot_right_y = crop_off_ratio_y * tf.random_uniform([], minval=0, maxval=1.0)


    tar_w_ratio = 1.0 - new_bot_right_x - new_top_left_x
    tar_h_ratio = 1.0 - new_bot_right_y - new_top_left_y

    target_height = tf.cast(tar_h_ratio * img_shape_list[0], tf.float32)
    target_width = tf.cast(tar_w_ratio * img_shape_list[1], tf.float32)
    image = tf.image.crop_to_bounding_box(image, offset_height=tf.cast(new_top_left_y * img_shape_list[0], tf.int32),
                                        offset_width=tf.cast(new_top_left_x * img_shape_list[1], tf.int32),
                                        target_height=tf.cast(tar_h_ratio * img_shape_list[0], tf.int32),
                                        target_width=tf.cast(tar_w_ratio * img_shape_list[1], tf.int32))

    # rotate image

    scale = tf.random_uniform([], minval=scale_range[0], maxval=scale_range[1])


    image = tf.image.resize_images(image,
                                 (tf.cast(target_height * scale, tf.int32), tf.cast(target_width * scale, tf.int32)))
    image = tf.image.resize_image_with_crop_or_pad(image, img_shape_list[0], img_shape_list[1])
    angle = rotation_angle*tf.random_uniform(shape=[],minval=-1.0,maxval=1.0,dtype=tf.float32)

    image = tf.contrib.image.rotate(image, angle)

    if do_flip:
        image = tf.image.flip_left_right(image)



    return image
