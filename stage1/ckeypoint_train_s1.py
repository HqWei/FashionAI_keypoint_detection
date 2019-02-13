from input_lib import data_input_mask as data_input
import tensorflow as tf
from net import fai_cpn_net as mnet
from stage1.config_s1 import Config
import os
import time


config = Config()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


slim = tf.contrib.slim


def train(category,steps):
    '''

    :param category: the category what you want to train
    :return:
    '''

    img_category = config.img_category
    category_classnum_dict = config.category_classnum_dict
    category_change_index = config.category_change_index
    batch_size = config.BATCH_SIZE
    lr = config.LEARNING_RATE
    lr_decay_rate = config.LR_DECAY_RATE
    lr_decay_step = config.LR_DECAY_STEP
    topk_dict = config.topk_dict
    batch_size_val = 8

    img_size=config.IMAGE_SIZE

    if category not in img_category:
        raise ValueError('wrong category')

    numclass = category_classnum_dict[category]

    # define data path
    train_data_path = '../data/tfrecord_s1/train/'+category+'.tfrecord'
    val_data_path = '../data/tfrecord_s1/val/'+category+'.tfrecord'
    log_path = 'logs/'+category
    weights_path = 'weights/'+category
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(val_data_path):
        raise ValueError("can't find val data path")
    if not os.path.exists(train_data_path):
        raise ValueError("can't find train data path")

    with tf.Graph().as_default():
        #with tf.device('/cpu:0'):
        (batch_x,batch_y,batch_pm)= data_input.read_batch(tfr_path=train_data_path,
                                                           numclass=numclass,
                                                           change_index=category_change_index[category],
                                                           argument=True,
                                                           img_size=img_size,
                                                           batch_size=batch_size)
        (batch_x_val,batch_y_val,batch_pm_val) = data_input.read_batch_val(tfr_path=val_data_path,
                                                                            numclass=numclass,
                                                                            change_index=category_change_index[category],
                                                                            argument=False,
                                                                            img_size=img_size,
                                                                            batch_size=batch_size_val)
        with tf.variable_scope('cpn_model'):
            model = mnet.CPN(numclass,batch_size)
            model.build_model(batch_x,True)
            model.build_loss_cpn(batch_y, batch_pm,lr, lr_decay_rate, lr_decay_step,top_k=topk_dict[category])
        with tf.variable_scope('cpn_model',reuse=True):
            model_val = mnet.CPN(numclass,batch_size_val)
            model_val.build_model(batch_x_val,False)
            model_val.build_loss_cpn(batch_y_val,batch_pm_val, lr, lr_decay_rate, lr_decay_step,top_k=topk_dict[category],val=True)




        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)



            saver = tf.train.Saver(max_to_keep=None)

            checkpoint_path = tf.train.latest_checkpoint(weights_path)
            if checkpoint_path is None:
                init = tf.global_variables_initializer()
                sess.run(init)

                print ('initialize from resnet_v1_101.ckpt')
                # remove some name
                def _removename(var):
                    return var.op.name.replace('cpn_model/', '')

                all_vars = slim.get_model_variables()
                var_to_restore = []
                for var in all_vars:
                    if 'resnet_v1_101' in var.op.name:
                        var_to_restore.append(var)
                    else:
                        continue
                var_to_restore = {_removename(var): var for var in var_to_restore}
                saver_part = tf.train.Saver(var_list=var_to_restore)
                saver_part.restore(sess, 'init_weights/resnet_v1_101.ckpt')

            else:
                saver.restore(sess, checkpoint_path)



            summary_writer = tf.summary.FileWriter(log_path,sess.graph)


            for i in range(steps):
                t1= time.time()
                _,gloss,reloss,reloss2,allloss,\
                global_steps,current_lr,summary= sess.run([   model.train_op,
                                                              model.global_loss,
                                                              model.refine_loss,
                                                              model.refine_loss2,
                                                              model.all_loss,
                                                              model.global_step,
                                                              model.lr,
                                                              model.loss_summary,
                                                                  ])
                summary_writer.add_summary(summary, global_steps)


                print('##========Iter {:>6d}========##'.format(global_steps))
                print('Current learning rate: {:.8f}'.format(current_lr))
                print('Traing time: {:.4f}'.format(time.time() - t1))
                print('gloss loss: {:>.6f}\n'.format(gloss))
                print('reloss loss2: {:>.6f}\n'.format(reloss2))
                print('reloss loss: {:>.6f}\n'.format(reloss))
                print('Total loss: {:>.6f}\n'.format(allloss))




                # save the val_loss value to choose which step to use for test
                if global_steps%50 ==0:
                    gloss_val,reloss_val,allloss_val,summary_val = sess.run([model_val.global_loss,
                                                                             model_val.refine_loss,
                                                                             model_val.all_loss,
                                                                             model_val.loss_summary])
                    summary_writer.add_summary(summary_val,global_steps)

                    print('********************************************************')
                    print('##========VAL Iter {:>6d}========##'.format(global_steps))
                    print('gloss loss: {:>.6f}\n\n'.format(gloss_val))
                    print('reloss loss: {:>.6f}\n\n'.format(reloss_val))
                    print('Total loss: {:>.6f}\n\n'.format(allloss_val))
                    print('********************************************************')
                if global_steps%1000 ==0:
                    saver.save(sess, weights_path+'/{}.ckpt'.format(global_steps))
                    print('\nModel checkpoint saved...\n')



            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    train('blouse',20000)

