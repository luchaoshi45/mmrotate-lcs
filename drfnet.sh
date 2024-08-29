#!/bin/bash

device=2,3

#config='oriented-rcnn-le90_r50_fpn_1x_dota_drfnet_rfla'                       # drfnet+rfla
#config='oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s'                             # baseline
config='oriented-rcnn-le90_r50_fpn_1x_dota_drfnet'                            # drfnet
#config='oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba'                       # mamba
#config='oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_rfla'                        # rfla
#config='oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla'                  # mamba+rfla
#config='oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_smoothgiou_twoStage_rfla'    # rfla+smoothgiou

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