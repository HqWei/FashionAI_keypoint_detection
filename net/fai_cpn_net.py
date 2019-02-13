import tensorflow as tf
from net import res101

slim = tf.contrib.slim

class CPN():

    def __init__(self,numclass,batch_size):
        self.numclass = numclass
        self.batch_size = batch_size
        self.wd = 5e-4
        self.output_shape = [96, 96]
        self.global_outs = []



    def build_model(self,inputs,is_training):
        batch_norm_params = {
            'decay':0.99,
            'epsilon':1e-9,
            'scale':True,
            'updates_collections':tf.GraphKeys.UPDATE_OPS,
            'is_training':is_training,
            'trainable': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(self.wd),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params
                            ):
            with slim.arg_scope([slim.conv2d_transpose],
                                activation_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(self.wd),
                                normalizer_fn=None
                                ):
                with slim.arg_scope([slim.batch_norm],**batch_norm_params):

                    with tf.variable_scope('resnet_v1_101') as sc50:
                        C2,C3,C4,C5= res101.build_res101(inputs)

                    with tf.variable_scope('fpn'):

                        P5 = slim.conv2d(C5, 256, [1,1],scope='c5p5')

                        p5up = tf.image.resize_bilinear(P5, (tf.shape(C4)[1], tf.shape(C4)[2]))
                        p5up = slim.conv2d(p5up, 256, [1, 1], activation_fn=None,normalizer_fn=None,scope='transpose_p5')
                        p4t = slim.conv2d(C4, 256,[1,1],padding='SAME',scope='c4p4' )
                        P4 = tf.add(p5up,p4t)

                        p4up = tf.image.resize_bilinear(P4, (tf.shape(C3)[1], tf.shape(C3)[2]))
                        p4up = slim.conv2d(p4up, 256, [1, 1], activation_fn=None,normalizer_fn=None,scope='transpose_p4')
                        p3t = slim.conv2d(C3, 256,[1,1],scope='c3p3' )
                        P3 = tf.add(p4up,p3t)

                        p3up = tf.image.resize_bilinear(P3, (tf.shape(C2)[1], tf.shape(C2)[2]))
                        p3up = slim.conv2d(p3up, 256, [1, 1], activation_fn=None,normalizer_fn=None,scope='transpose_p3')
                        p2t = slim.conv2d(C2, 256, [1,1],scope='c2p2')
                        P2 = tf.add(p3up,p2t)

                        self.p5out = slim.conv2d(P5, self.numclass, [3, 3], scope='p5out', activation_fn=None,normalizer_fn=None)
                        self.global_outs.append(
                            tf.image.resize_bilinear(self.p5out, (tf.shape(C2)[1], tf.shape(C2)[2])))
                        self.p4out = slim.conv2d(P4, self.numclass, [3, 3], scope='p4out', activation_fn=None,normalizer_fn=None)
                        self.global_outs.append(
                            tf.image.resize_bilinear(self.p4out, (tf.shape(C2)[1], tf.shape(C2)[2])))
                        self.p3out = slim.conv2d(P3, self.numclass, [3, 3], scope='p3out', activation_fn=None,normalizer_fn=None)
                        self.global_outs.append(
                            tf.image.resize_bilinear(self.p3out, (tf.shape(C2)[1], tf.shape(C2)[2])))
                        self.p2out = slim.conv2d(P2, self.numclass, [3, 3], scope='p2out', activation_fn=None,normalizer_fn=None)
                        self.global_outs.append(
                            tf.image.resize_bilinear(self.p2out, (tf.shape(C2)[1], tf.shape(C2)[2])))

                        self.pout = [self.p2out, self.p3out, self.p4out, self.p5out]

                    with tf.variable_scope('refine'):
                        P5 = res101.bottleneck(inputs=P5, depth=256, depth_bottleneck=128, stride=1, scope='p5block1')
                        P5 = res101.bottleneck(inputs=P5, depth=256, depth_bottleneck=128, stride=1, scope='p5block2')
                        P5 = res101.bottleneck(inputs=P5, depth=256, depth_bottleneck=128, stride=1, scope='p5block3')
                        P5U = tf.image.resize_bilinear(P5, (tf.shape(C2)[1], tf.shape(C2)[2]))


                        P4 = res101.bottleneck(inputs=P4, depth=256, depth_bottleneck=128, stride=1, scope='p4block1')
                        P4 = res101.bottleneck(inputs=P4, depth=256, depth_bottleneck=128, stride=1, scope='p4block2')
                        P4U = tf.image.resize_bilinear(P4, (tf.shape(C2)[1], tf.shape(C2)[2]))

                        P3 = res101.bottleneck(inputs=P3, depth=256, depth_bottleneck=128, stride=1, scope='p3block1')
                        P3U = tf.image.resize_bilinear(P3, (tf.shape(C2)[1], tf.shape(C2)[2]))

                        P2U = P2

                        PC = tf.concat([P5U,P4U,P3U,P2U],axis=-1)

                        PC = res101.bottleneck(inputs=PC, depth=256, depth_bottleneck=128, stride=1, scope='pcblock')
                        self.finalout = slim.conv2d(PC, self.numclass, [3, 3], scope='pcout',activation_fn=None,normalizer_fn=None)
        return tf.contrib.framework.get_variables(sc50)

    def build_loss_cpn(self, gt_heatmap, pm_mask,lr, lr_decay_rate, lr_decay_step, top_k=1, val=False):

        self.all_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        gt_heatmap_cal = gt_heatmap

        def ohkm(loss, top_k):
            ohkm_loss = 0.
            for i in range(self.batch_size):
                sub_loss = loss[i]
                topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
                tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i))  # can be ignore ???
                ohkm_loss += tf.reduce_sum(tmp_loss) / top_k
            ohkm_loss /= self.batch_size
            return ohkm_loss

        self.global_loss = 0.
        with tf.variable_scope('global_loss'):
            for x in range(len(self.global_outs)):

                all_class_loss = tf.reduce_mean(tf.square(self.global_outs[x] - gt_heatmap_cal), (1, 2)) *pm_mask
                self.global_loss += tf.reduce_sum(all_class_loss) / tf.reduce_sum(pm_mask)
            self.global_loss /= len(self.global_outs)
            self.global_loss /= 2.

        with tf.variable_scope('refine_loss'):
            refine_loss = tf.reduce_mean(tf.square(self.finalout - gt_heatmap_cal), (1, 2))
            refine_loss = refine_loss *pm_mask
            refine_loss /= 2.
            self.refine_loss = ohkm(refine_loss, top_k)

            all_class_loss2 = tf.reduce_mean(tf.square(self.finalout - gt_heatmap_cal), (1, 2))*pm_mask
            self.refine_loss2 = tf.reduce_sum(all_class_loss2) / tf.reduce_sum(pm_mask)
            self.refine_loss2 /=2.
        self.regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        with tf.variable_scope('all_loss'):
            self.all_loss = self.global_loss + self.refine_loss +self.regularization_loss

        if not val:
            with tf.variable_scope('train'):
                self.global_step = tf.contrib.framework.get_or_create_global_step()

                self.lr = tf.train.exponential_decay(self.learning_rate,
                                                     global_step=self.global_step,
                                                     decay_rate=self.lr_decay_rate,
                                                     decay_steps=self.lr_decay_step)

                opt = tf.train.AdamOptimizer(learning_rate=self.lr)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([tf.group(*update_ops)]):
                    self.train_op = slim.learning.create_train_op(self.all_loss, opt, global_step=self.global_step)
        if val:
            self.loss_summary = tf.summary.scalar('total loss_val', self.all_loss)
        else:
            self.loss_summary = tf.summary.scalar('total loss', self.all_loss)
























