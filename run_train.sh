### Individual naming/format
# NODDI: [32, 35, ..., 49, 50]
# Oddball: [sub001, sub002, ..., sub016, sub017]
# CNEPFL: [sub-02, sub-04, ..., sub-24, sub-26]

## To train E2fNet
CUDA_VISIBLE_DEVICES=0 python train_E2fNet.py \
--dataset NODDI \
--test_ids 43 \
--fmri_channel 30 \
--exp_root /home/Experiments/EEG2fMRI \
--batch_size 32 \
--num_epochs 50 \
--lr 0.001

# ## To train E2fGAN
# CUDA_VISIBLE_DEVICES=0 python train_E2fGAN.py \
# --dataset NODDI \
# --test_ids 43 \
# --fmri_channel 30 \
# --exp_root /home/Experiments/EEG2fMRI \
# --batch_size 32 \
# --num_epochs 50 \
# --lr 0.001