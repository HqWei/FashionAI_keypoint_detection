import numpy as np
columns  =[   'image_id','image_category',
              'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
              'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
              'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
              'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
              'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
              'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'
]

def offval(category,step):
    anofile = open('../../data/train_all/val_split/'+category+'_val.txt').read().splitlines()
    predfile = open('/mnt/sda1/don/documents/fashion_ai/cpn/result_make/'+category+'_'+str(step)+'.csv').read().splitlines()[1:]

    all_point_valid = 0
    all_point = 0.
    for i in range(len(anofile)):
        line_ano = anofile[i].split(',')
        line_pre = predfile[i].split(',')

        cate = line_ano[1]
        if cate == 'blouse' or cate == 'outwear' or cate == 'dress':
            n1 = np.array((line_ano[7].split('_')),dtype=np.int32)
            n2 = np.array((line_ano[8].split('_')), dtype=np.int32)
            if n1[2]==-1 or n2[2]==-1:
                continue
            norm = np.sqrt(np.square(n1[0]-n2[0])+np.square(n1[1]-n2[1]))

        elif cate=='skirt' or cate=='trousers':
            n1 = np.array((line_ano[17].split('_')),dtype=np.int32)
            n2 = np.array((line_ano[18].split('_')), dtype=np.int32)
            if n1[2]==-1 or n2[2]==-1:
                continue
            norm = np.sqrt(np.square(n1[0]-n2[0])+np.square(n1[1]-n2[1]))
        else:
            raise ValueError('wrong type')

        for j in range(24):
            p_ano = np.array((line_ano[j+2].split('_')),dtype=np.int32)
            p_pre = np.array((line_pre[j + 2].split('_')), dtype=np.int32)
            if p_ano[2]==1:
                all_point_valid+=1
                point_d = np.sqrt(np.square(p_ano[0] - p_pre[0]) + np.square(p_ano[1] - p_pre[1])) / norm
                all_point +=point_d
            else:
                continue
    #print('score: {:.2f}%'.format((all_point / all_point_valid) * 100))
    return (all_point / all_point_valid) * 100

# the second
#print (offval('blouse',35000))