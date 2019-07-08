import glob
import os
import sys

os.system('mkdir -p sat/train')

train_img_list = glob.glob("/qfs/projects/sgdatasc/spacenet/Vegas_processed_train/annotations/RGB-PanSharpen/*.tif")


for i, img in enumerate(train_img_list[:400]):
    fname = '.'.join(os.path.basename(img).split('.')[:-1])
    targ = "/qfs/projects/sgdatasc/spacenet/Vegas_processed_train/" + \
        "annotations/annotations/{}segobj.tif".format(fname)
    os.system('cp {} sat/train/img{:03d}.tif'.format(img, i))
    os.system('cp {} sat/train/target{:03d}.tif'.format(targ, i))

os.system('mkdir -p sat/test')

test_img_list = glob.glob("/qfs/projects/sgdatasc/spacenet/Vegas_processed_test/annotations/RGB-PanSharpen/*.tif")

for i, img in enumerate(test_img_list[:100]):
    fname = '.'.join(os.path.basename(img).split('.')[:-1])
    targ = "/qfs/projects/sgdatasc/spacenet/Vegas_processed_test/" + \
        "annotations/annotations/{}segobj.tif".format(fname)
    os.system('cp {} sat/test/img{:03d}.tif'.format(img, i))
    os.system('cp {} sat/test/target{:03d}.tif'.format(targ, i))
