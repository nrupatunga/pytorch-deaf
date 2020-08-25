DATA_PATH='/home/nthere/2020/pytorch-deaf/data/DIV_superres/hdf5/'
#DATA_PATH='/media/nthere/datasets/deaf/L0/'

CUDA_VISIBLE_DEVICES=0 python train.py --data_dir $DATA_PATH
