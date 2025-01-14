import numpy as np
import scipy.io as scio

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    l = len(data)
    data = sorted(data)
    for i in data:
        s = str(i).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

data = [[57, -1], [262, 1], [18, -1], [494, 1], [516, 1], [240, 1], [368, 1], [636, 1], [434, 1], [35, 1],
        [257, 1], [466, 1], [42, 1], [431, 1], [341, 1], [252, 1], [525, 1], [517, 1], [381, 1], [2, 1],
        [470, 1], [531, 1], [437, 1], [378, 1], [340, 1], [118, 1], [108, 1], [256, 1], [532, 1], [267, 1],
        [673, 1], [690, 1], [327, 1], [164, 1], [290, 1], [43, 1], [229, 1], [45, 1], [600, 1], [438, 1],
        [30, 1], [613, 1], [374, 1], [316, 1], [276, 1], [273, 1], [648, 1], [189, 1], [40, 1], [245, 1],
        [74, 1], [521, 1], [383, 1], [305, 1], [224, 1], [678, 1], [390, 1], [138, 1], [488, 1], [238, 1],
        [515, 1], [249, 1], [71, 1], [482, 1], [691, 1], [92, 1], [699, 1], [272, 1], [671, 1], [436, 1],
        [51, 1], [624, 1], [175, 1], [175, 1], [548, 1], [329, 1], [687, 1], [270, 1], [225, 1], [518, 1],
        [28, 1], [25, 1], [32, 1], [345, 1], [153, 1], [352, 1], [419, 1], [298, 1], [435, 1], [513, 1],
        [291, 1], [538, 1], [194, 1], [590, 1], [660, 1], [242, 1], [414, 1], [665, 1], [285, 1], [271, 1],
        [184, 1], [235, 1], [31, 1], [65, 1], [657, 1], [19, 1], [103, 1], [679, 1], [534, 1], [441, 1],
        [556, 1], [478, 1], [605, 1], [88, 1], [110, 1], [322, 1], [562, 1], [677, 1], [15, 1], [258, 1],
        [23, 1], [370, 1], [420, 1], [147, 1], [115, 1], [210, 1], [492, 1], [666, 1], [680, 1], [621, 1],
        [11, 1], [428, 1], [321, 1], [608, 1], [5, 1], [391, 1], [536, 1], [365, 1], [199, 1], [152, 1],
        [575, 1], [119, 1], [694, 1], [313, 1], [260, 1], [179, 1], [283, 1], [323, 1], [131, 1], [545, 1],
        [551, 1], [579, 1], [543, 1], [396, 1], [599, 1], [127, 1], [136, 1], [464, 1], [612, 1], [592, 1],
        [58, 1], [13, 1], [393, 1], [682, 1], [213, 1], [91, 1], [204, 1], [618, 1], [421, 1], [156, 1],
        [635, 1], [61, 1], [426, 1], [539, 1], [173, 1], [105, 1], [568, 1], [151, 1], [688, 1], [379, 1],
        [503, 1], [124, 1], [387, 1], [60, 1], [317, 1], [417, 1], [601, 1], [82, 1], [69, 1], [583, 1],
        [281, 1], [163, 1], [523, 1], [634, 1], [411, 1], [144, 1], [41, 1], [637, 1], [26, 1], [26, 1],
        [476, 1], [3, 1], [629, 1], [50, 1], [98, 1], [6, 1], [623, 1], [430, 1], [649, 1], [357, 1], [508, 1],
        [696, 1], [36, 1], [126, 1], [491, 1], [142, 1], [254, 1], [135, 1], [264, 1], [582, 1], [448, 1],
        [17, 1], [454, 1], [221, 1], [496, 1], [243, 1], [306, 1], [655, 1], [67, 1], [372, 1], [113, 1],
        [315, 1], [315, 1], [302, 1], [157, 1], [416, 1], [7, 1], [16, 1], [358, 1], [646, 1], [554, 1],
        [632, 1], [644, 1], [457, 1], [386, 1], [9, 1], [162, 1], [99, 1], [461, 1], [46, 1], [288, 1],
        [49, 1], [574, 1], [452, 1], [94, 1], [191, 1], [346, 1], [62, 1], [198, 1], [570, 1], [183, 1],
        [309, 1], [552, 1], [112, 1], [526, 1], [211, 1], [497, 1], [188, 1], [472, 1], [96, 1], [558, 1],
        [348, 1], [121, 1], [177, 1], [353, 1], [617, 1], [650, 1], [445, 1], [279, 1], [670, 1], [220, 1],
        [662, 1], [54, 1], [672, 1], [293, 1], [195, 1], [512, 1], [143, 1], [641, 1], [607, 1], [230, 1],
        [132, 1], [549, 1], [639, 1], [366, 1], [209, 1], [520, 1], [520, 1], [483, 1], [559, 1], [403, 1],
        [645, 1], [589, 1], [577, 1], [395, 1], [170, 1], [77, 1], [38, 1], [564, 1], [53, 1], [410, 1],
        [663, 1], [155, 1], [48, 1], [362, 1], [311, 1], [86, 1], [234, 1], [578, 1], [39, 1], [363, 1],
        [75, 1], [356, 1], [187, 1], [336, 1], [222, 1], [500, 1], [102, 1], [380, 1], [501, 1], [375, 1],
        [217, 1], [263, 1], [619, 1], [72, 1], [606, 1], [248, 1], [79, 1], [171, 1], [185, 1], [168, 1],
        [226, 1], [622, 1], [382, 1], [533, 1], [219, 1], [106, 1], [297, 1], [328, 1], [616, 1], [63, 1],
        [586, 1], [486, 1], [469, 1], [192, 1], [377, 1], [373, 1], [484, 1], [668, 1]]

filename = './fish_prediction_result.txt'

text_save(filename, data)