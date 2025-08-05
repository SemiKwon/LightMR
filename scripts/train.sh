#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$PYTHONPATH:/mnt/aix23207/MASTC

save_dir=/mnt/aix23207/MASTC/checkpoint_final_ver
deepspeed_config=/mnt/aix23207/MASTC/ds_config.json
video_dir=/mnt/aix23207/Charades_v1_480
train_txt=/mnt/aix23207/charades_sta_train.txt
test_txt=/mnt/aix23207/charades_sta_test.txt
batch_size=2
num_epochs=50
patience=5

PYTHONPATH=$PYTHONPATH:. python train_run.py \
  --save_dir ${save_dir} \
  --deepspeed_config ${deepspeed_config} \
  --video_dir ${video_dir} \
  --train_txt ${train_txt} \
  --test_txt ${test_txt} \
  --batch_size ${batch_size} \
  --num_epochs ${num_epochs} \
  --patience ${patience}
