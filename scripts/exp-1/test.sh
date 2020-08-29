CKPT_PATH='./ckpt/deaf_deaf__ckpt_epoch_8.ckpt'
IMG_PATH='/home/nthere/2020/L0-Smoothing/src/images/flower2.png'

python test.py \
	--ckpt_path $CKPT_PATH \
	--img_path $IMG_PATH
