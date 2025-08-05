#!/bin/bash

# 사용할 GPU 설정 (0번 GPU 사용)
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$PYTHONPATH:/mnt/aix23207/MASTC

# 실험 설정
save_dir=/mnt/aix23207/MASTC/inference_3
weight_dir=/mnt/aix23207/MASTC/checkpoint
ds_config_path=/mnt/aix23207/MASTC/ds_config.json
video_dir=/mnt/aix23207/Charades_v1_480
train_txt=/mnt/aix23207/charades_sta_train.txt
test_txt=/mnt/aix23207/charades_sta_test.txt
batch_size=4

# 실행
PYTHONPATH=$PYTHONPATH:. python eval_run.py \
  --save_dir ${save_dir} \
  --weight_dir ${weight_dir} \
  --ds_config_path ${ds_config_path} \
  --video_dir ${video_dir} \
  --train_txt ${train_txt} \
  --test_txt ${test_txt} \
  --batch_size ${batch_size}