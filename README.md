# MobileFAN: transferring deep hidden representation for face alignment

# Requirements:
Python=2.7 and Pytorch=0.4.0

Thanks to Simple Baselines for Human Pose Estimation (https://github.com/microsoft/human-pose-estimation.pytorch).

# Training
e.g. for cofw

python /example/Img_22de_128_pair_pixel_d1d2d3.py --checkpoint ./checkpoint/Img_22de_128_pair_pixel_d1d2d3/model1/ --epochs 80 --train-batch 8 --test-batch 8 --learning-rate 0.001 -t_pixel 0.01 -t_pair 0.01 --schedule 30 50

# Citation
If you use our code in your research or wish to refer to the baseline results, please cite our paper:
```
@article{zhao2020mobilefan,
  title={MobileFAN: transferring deep hidden representation for face alignment},
  author={Zhao, Yang and Liu, Yifan and Shen, Chunhua and Gao, Yongsheng and Xiong, Shengwu},
  journal={Pattern Recognition},
  volume={100},
  pages={107114},
  year={2020},
  publisher={Elsevier}
}
```
