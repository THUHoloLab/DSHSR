# DSHSR
# Aperture Array Based Self-Supervised Hyperspectral Super-Resolution Imaging
# Training
```
python train.py --model_ME_path model_ME_path --train_file train_file --gray_max gray_max --offset_max offset_max 
```
# Test
```
python test_synthetic.py --model_path model_path --test_size 448 --test_file test_file --gray_max gray_max --offset_max offset_max --save_SR save_SR --save_LR save_LR --save_GT save_GT
```
```
python test_real.py --model_path model_path --test_size 448 --test_file test_file --gray_max gray_max --save_SR save_SR --save_LR save_LR

