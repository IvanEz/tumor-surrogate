#!/bin/bash

python3 main.py --is_train=False --load_path="" --test_batch_size=1 --is_3d=True --dataset='tumor_mparam' --res_x=64 --res_y=64 --res_z=64 --batch_size=4 --num_worker=1 --num-conv 4 --use_curl False --arch 'alternative' --phys_loss False --gpu_id="4" --data_dir="" --inf_save=""
