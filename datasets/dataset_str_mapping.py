'''
Dictionary to map a dataset_str to the location of the data file containing
gt labels, annotations, etc.
'''

MAPPING = {
    'cocodots_train': './data/coco_dots_0_0_train2017.json',
    'cocodots_val': './data/coco_dots_0_0_val2017.json',
    'cocodots_val_mini': './data/coco_dots_0_0_val2017_mini.json',
    'mazes_train': './data/mazes_train.json',
    'mazes_val': './data/mazes_val.json',
    'F3_train': './data/float_3seg_train.json',
    'F3_val': './data/float_3seg_val.json',
    'FN_train': './data/float_varseg_train.json',
    'FN_val': './data/float_varseg_val.json',
    'BN_train': './data/binary_varseg_train.json',
    'BN_val': './data/binary_varseg_val.json',
    'BN_bal_train': './data/binary_bal_varseg_train.json',
    'BN_bal_val': './data/binary_bal_varseg_val.json',

}