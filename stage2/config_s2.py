# define a unity config class to avoid subdefine which may cause conflicts
class Config():
    IMAGE_SIZE = 512
    BATCH_SIZE = 16

    # learning param
    LEARNING_RATE = 0.00005  # 0.0001-0.00005(20000)
    LR_DECAY_RATE = 0.5
    LR_DECAY_STEP = 20000

    img_category = ['blouse', 'skirt', 'outwear', 'dress', 'trousers']

    blouse_labels = ["neckline_left", "neckline_right", "center_front", "shoulder_left", "shoulder_right",
                     "armpit_left", "armpit_right", "cuff_left_in", "cuff_left_out", "cuff_right_in",
                     "cuff_right_out", "top_hem_left", "top_hem_right"]

    outwear_labels = ["neckline_left", "neckline_right", "shoulder_left", "shoulder_right",
                      "armpit_left", "armpit_right", "waistline_left", "waistline_right", "cuff_left_in",
                      "cuff_left_out", "cuff_right_in", "cuff_right_out", "top_hem_left", "top_hem_right"]

    dress_labels = ["neckline_left", "neckline_right", "center_front", "shoulder_left", "shoulder_right",
                    "armpit_left", "armpit_right", "waistline_left", "waistline_right", "cuff_left_in",
                    "cuff_left_out", "cuff_right_in", "cuff_right_out", "hemline_left", "hemline_right"]

    bod_labels = ["neckline_left", "neckline_right", "center_front", "shoulder_left", "shoulder_right",
                  "armpit_left", "armpit_right", "waistline_left", "waistline_right", "cuff_left_in",
                  "cuff_left_out", "cuff_right_in", "cuff_right_out", "top_hem_left", "top_hem_right", "hemline_left",
                  "hemline_right"]

    skirt_labels = ["waistband_left", "waistband_right", "hemline_left", "hemline_right"]
    trousers_labels = ["waistband_left", "waistband_right", "crotch", "bottom_left_in", "bottom_left_out",
                       "bottom_right_in", "bottom_right_out"]

    st_labels = ["waistband_left", "waistband_right", "hemline_left", "hemline_right",
                 "crotch",
                 "bottom_left_in", "bottom_left_out",
                 "bottom_right_in", "bottom_right_out"
                 ]

    all_labels = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                  'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left',
                  'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                  'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left',
                  'waistband_right', 'hemline_left', 'hemline_right', 'crotch',
                  'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    category_label_dict = {'blouse': blouse_labels,
                           'skirt': skirt_labels,
                           'outwear': outwear_labels,
                           'dress': dress_labels,
                           'trousers': trousers_labels,
                           'bod': bod_labels,
                           'st': st_labels}

    category_classnum_dict = {'blouse': 13,
                              'skirt': 4,
                              'outwear': 14,
                              'dress': 15,
                              'trousers': 7,
                              'bod': 17,
                              'st': 9}

    topk_dict = {'blouse': 6,
                 'skirt': 3,
                 'outwear': 7,
                 'dress': 7,
                 'trousers': 5,
                 'bod': 8,
                 'st': 3}
    #
    category_change_index = {
        'blouse': [[0, 3, 5, 7, 8, 11],
                   [1, 4, 6, 9, 10, 12]],
        'skirt': [[0, 2],
                  [1, 3]],
        'outwear': [[0, 2, 4, 6, 8, 9, 12],
                    [1, 3, 5, 7, 10, 11, 13]],
        'dress': [[0, 3, 5, 7, 9, 10, 13],
                  [1, 4, 6, 8, 11, 12, 14]],
        'trousers': [[0, 3, 4],
                     [1, 5, 6]],

        # no use in the competition
        'bod': [[0, 3, 5, 7, 9, 10, 13, 15],
                [1, 4, 6, 8, 11, 12, 14, 16]],
        'st': [[0, 2, 5, 6],
               [1, 3, 7, 8]]
    }
