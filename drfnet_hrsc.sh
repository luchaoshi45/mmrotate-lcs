#!/bin/bash

device=1,2

#config='oriented-rcnn-le90_r50_fpn_3x_hrsc_drfnet_rfla'
#config='oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s'
config='oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba'
#config='oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_rfla'
#config='oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_mamba_rfla'
#config='oriented-rcnn-le90_r50_fpn_3x_hrsc_lsk_s_smoothgiou_twoStage_rfla'

pth_name='epoch_12'

work_dir=$config
IFS=',' read -ra numbers <<< "$device"
num_device=${#numbers[@]}

count=$#
if [ $count -eq 0 ]; then
  CUDA_VISIBLE_DEVICES=$device \
  nohup \
  tools/dist_train.sh \
  configs/drfnet/$config.py \
  $num_device \
  --cfg-options work_dir='./work_dirs/'$work_dir \
  2>&1 | tee -a work_dirs/.atrain_add.log | tee work_dirs/.atrain.log &
elif [ $count -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=$device \
  tools/dist_test.sh \
  configs/drfnet/$config.py \
  work_dirs/$config/$pth_name.pth \
  $num_device
else
    echo "多输入参数错误。"
fi