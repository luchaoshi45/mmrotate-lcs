#!/bin/bash
device_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "\e[32mdevice_num: $device_num\e[0m"
configs=(
    "oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_rfla"
    "oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba"
    "oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s"
)
echo -e "\e[32mconfigs: ${configs[@]}\e[0m"

for config in "${configs[@]}"; do
    if [[ -f "tools/dist_train.sh" && -f "tools/dist_test.sh" && -f "configs/drfnet/${config}.py" ]]; then
        tools/dist_train.sh "configs/drfnet/${config}.py" $device_num
        tools/dist_test.sh "configs/drfnet/${config}.py" "work_dirs/${config}/epoch_12.pth" $device_num
    else
        echo -e "\e[31mRequired files for ${config} are missing.\e[0m"
    fi
done


# rm -rf mmrotate-lcs
# git clone https://gitclone.com/github.com/luchaoshi45/mmrotate-lcs.git
# cd mmrotate-lcs
# chmod -R 777 *
# pip install -v -e .
# bash task.sh