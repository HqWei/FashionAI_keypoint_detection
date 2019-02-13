import numpy as np
import cv2
import math
import matplotlib.pyplot as plt





def get_location_cpn(stage_heatmap,input_img_size):

    last_heatmap = stage_heatmap[0, :, :, :]
    last_heatmap = cv2.resize(last_heatmap, (input_img_size, input_img_size))
    num_class = stage_heatmap[-1].shape[-1]


    joint_coord_set = np.zeros((num_class, 2))



    for joint_num in range(num_class):

        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (input_img_size, input_img_size))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

    return np.array(joint_coord_set,dtype=np.int32)


def get_location_cpn_s(stage_heatmap,input_img_size):

    last_heatmap = stage_heatmap
    last_heatmap = cv2.resize(last_heatmap, (input_img_size, input_img_size))
    num_class = stage_heatmap.shape[-1]


    joint_coord_set = np.zeros((num_class, 2))



    for joint_num in range(num_class):

        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (input_img_size, input_img_size))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

    return np.array(joint_coord_set,dtype=np.int32)

def get_location_cpn_s2(stage_heatmap,input_img_size):

    last_heatmap = stage_heatmap
    #last_heatmap = cv2.resize(last_heatmap, (input_img_size, input_img_size))
    num_class = stage_heatmap.shape[-1]


    joint_coord_set = np.zeros((num_class, 2))

    for w in range(num_class):
        last_heatmap[:, :, w] /= np.amax(last_heatmap[:,:,w])
        last_heatmap[:,:,w] = cv2.GaussianBlur(last_heatmap[:,:,w], (3, 3), 0)

    for joint_num in range(num_class):
        x = 0.
        y = 0.
        for z in range(10):
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (last_heatmap.shape[0], last_heatmap.shape[1]))
            last_heatmap[joint_coord[0],joint_coord[1],joint_num] = 0.
            y += joint_coord[0]
            x += joint_coord[1]
        x /=10.
        y /=10.
        joint_coord_set[joint_num, :] = [y*4, x*4]

    return np.array(joint_coord_set,dtype=np.int32)


def get_location_cpn_n_scale(stage_heatmap):


    num_class = stage_heatmap.shape[-1]
    output_shape = stage_heatmap.shape[:2]
    res = stage_heatmap.transpose(2,0,1)


    for w in range(num_class):
        res[w] /= np.amax(res[w])
    border = 10
    dr = np.zeros((num_class, output_shape[0] + 2 * border, output_shape[1] + 2 * border))
    dr[:, border:-border, border:-border] = res[:num_class].copy()

    for w in range(num_class):
        dr[w] = cv2.GaussianBlur(dr[w], (3, 3), 0)

    joint_coord_set = np.zeros((num_class, 2))
    for w in range(num_class):
        lb = dr[w].argmax()
        y, x = np.unravel_index(lb, dr[w].shape)
        dr[w, y, x] = 0
        lb = dr[w].argmax()
        py, px = np.unravel_index(lb, dr[w].shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, output_shape[1] - 1))
        y = max(0, min(y, output_shape[0] - 1))

        joint_coord_set[w, :] = [(y-10.) * 4, (x-10.) * 4]
    return np.array(joint_coord_set, dtype=np.int32)

def get_location_cpn_n(stage_heatmap):


    num_class = stage_heatmap.shape[-1]
    output_shape = stage_heatmap.shape[:2]
    res = stage_heatmap.transpose(2,0,1)


    for w in range(num_class):
        res[w] /= np.amax(res[w])
    border = 10
    dr = np.zeros((num_class, output_shape[0] + 2 * border, output_shape[1] + 2 * border))
    dr[:, border:-border, border:-border] = res[:num_class].copy()

    for w in range(num_class):
        dr[w] = cv2.GaussianBlur(dr[w], (3, 3), 0)

    joint_coord_set = np.zeros((num_class, 2))
    for w in range(num_class):
        lb = dr[w].argmax()
        y, x = np.unravel_index(lb, dr[w].shape)
        dr[w, y, x] = 0
        lb = dr[w].argmax()
        py, px = np.unravel_index(lb, dr[w].shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, output_shape[1] - 1))
        y = max(0, min(y, output_shape[0] - 1))

        joint_coord_set[w, :] = [y * 4, x * 4]
    return np.array(joint_coord_set, dtype=np.int32)


def get_location(stage_heatmap,input_img_size):

    last_heatmap = stage_heatmap[-1][0, :, :, :-1]
    last_heatmap = cv2.resize(last_heatmap, (input_img_size, input_img_size))
    num_class = stage_heatmap[-1].shape[-1]


    joint_coord_set = np.zeros((num_class-1, 2))



    for joint_num in range(num_class-1):

        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (input_img_size, input_img_size))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

    return np.array(joint_coord_set,dtype=np.int32)

def restore_location(ori_img_shape,label_output,scale,start_index):
    height_n = ori_img_shape[0]
    width_n = ori_img_shape[1]

    if height_n<=width_n:
        label_scale = [[sl[0]-start_index, sl[1]] for sl in label_output]
    else:

        label_scale = [[sl[0], sl[1]-start_index] for sl in label_output]
    label_scale = [[int(math.ceil(sl[0] / scale)), int(math.ceil(sl[1] / scale))] for sl in label_scale]
    return label_scale


def make_for_input(img,re_size):


    img_scale = np.ones([re_size,re_size,3],dtype=np.uint8)*128
    height = img.shape[0]
    width = img.shape[1]
    scale = float(re_size)/max(height,width)
    img_ = img.copy()
    img2=cv2.resize(img_,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    height_n = img2.shape[0]
    width_n = img2.shape[1]

    start_index = int(re_size/2-math.ceil(min(height_n,width_n)/2))
    if height_n<=width_n:
        img_scale[start_index:start_index+height_n,:,:]=img2
    else:
        img_scale[:,start_index:start_index+width_n,:]=img2


    return img_scale,scale,start_index


# sub_colors = [[0.0, 0.66, 1.0],
#               [0.0, 0.0, 1.0],
#               [1.0, 0.0, 0.0],
#               [0.0, 1.0, 0.0],
#               [1.0, 0.66, 0.0],
#               [0.0, 1.0, 0.66],
#               [0.66, 0.0, 1.0],
#               [1.0, 0.0, 0.66],
#               [0.0, 0.0, 0.0],
#               [0.0,0.33,1.0],
#               [0.5,0.22,0.33],
#               [0.33,0.66,0.33],
#               [0.88,0.77,0.33]]


def visualize_result(img_toshow,location):





    for joint_num in range(len(location)):

        joint_coord = location[joint_num]

        cv2.circle(img_toshow, center=(joint_coord[1], joint_coord[0]), radius=3, color=[0,255,0], thickness=-1)
        cv2.putText(img_toshow,str(joint_num+1),(joint_coord[1], joint_coord[0]),cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,0],2)

    img_toshow = img_toshow.astype(np.uint8)
    plt.imshow(img_toshow)
    plt.show()

def visualize_result2(img_toshow,location):

    for joint_num in range(len(location)):

        joint_coord = location[joint_num,:]

        cv2.circle(img_toshow, center=(joint_coord[1], joint_coord[0]), radius=3, color=[255,0,0], thickness=-1)
        cv2.putText(img_toshow,str(joint_num+1),(joint_coord[1], joint_coord[0]),cv2.FONT_HERSHEY_SIMPLEX,0.8,[255,0,0],2)

    img_toshow = img_toshow.astype(np.uint8)
    plt.imshow(img_toshow)
    plt.show()

def visualize_result_gt(img_toshow,location):

    for joint_num in range(len(location)):

        joint_coord = location[joint_num,:]

        cv2.circle(img_toshow, center=(joint_coord[0], joint_coord[1]), radius=3, color=[0,255,0], thickness=-1)
        cv2.putText(img_toshow,str(joint_num+1),(joint_coord[0], joint_coord[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,[0,255,0],2)

    img_toshow = img_toshow.astype(np.uint8)
    plt.imshow(img_toshow)
    plt.show()

def visualize_result_ori(test_img, stage_heatmap_np,num_class):

    last_heatmap = stage_heatmap_np[-1][0, :, :, :-1]

    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))


    joint_coord_set = np.zeros((num_class-1, 2))



    for joint_num in range(num_class-1):

        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]



        cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=[255,0,0], thickness=-1)
        cv2.putText(test_img,str(joint_num),(joint_coord[1], joint_coord[0]),cv2.FONT_HERSHEY_SIMPLEX,0.5,[255,0,0],1)


    test_img = test_img.astype(np.uint8)
    plt.imshow(test_img)
    plt.show()

    return test_img