#CKPT_PATH='./deaf/deaf_epoch=19.ckpt'
CKPT_PATH='./new_arch_1/deaf_deaf__ckpt_epoch_16.ckpt'
#IMG_PATH='/home/nthere/2020/vcnn_double-bladed/applications/deep_edge_aware_filters/images/3.png'
IMG_PATH='/home/nthere/2020/L0-Smoothing/src/images/basketball.png'

python test_new.py \
	--ckpt_path $CKPT_PATH \
	--img_path $IMG_PATH
