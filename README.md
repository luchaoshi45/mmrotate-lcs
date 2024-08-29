# MMrotate-lcs

## 1 Install

### Env
```shell
xjtu@liyaochen122
xjtu@liyaochen2024

su root
conda create --name lcs_mmrotate-lcs --no-default-packages python=3.8
# conda remove --name lcs_mmrotate-lcs --all
# rm -r /data1/anaconda3/envs/lcs_mmrotate-lcs

su lyc
conda activate lcs_mmrotate-lcs
```

### Requirements
```shell
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install mmengine==0.10.4
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmpretrain==1.2.0
pip install triton==2.0.0
pip install transformers==4.42.4
pip install timm
# github down
pip install causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## 2 Run

### Train
```shell
tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_drfnet.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_rfla.py 2

tools/dist_train.sh configs/drfnet/oriented-rcnn-le90_r50_fpn_1x_dota_lsk_s_mamba_rfla.py 2
```