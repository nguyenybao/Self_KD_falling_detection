# Cải tiến mô hình..............

Sinh viên thực hiện Mai Nguyen, Bao Yen Nguyen

## Tổng quan đề tài 
mục đích là gì, hướng tiếp cận ra xaooooooooo
Self-knowledge distillation, the related idea to knowledge distillation, is a novel approach to avoid training a large teacher network. 
In our work, we propose an efficient self-knowledge distillation approach for falling detection. In our approach, the network shares and learns the knowledge distilled via embedded vectors from two different views of a data point. Moreover, we also present the lightweights yet robust network to address this task based on (2+1)D convolution. Our proposed network uses only 0.1M parameters that reduce hundreds of times compared to other deep networks. To evaluate the effectiveness of our proposed approach, two standard datasets such as the FDD and URFD datasets, have been experimented. The results illustrated state-of-the-art performance and outperformed that compared to independent training. Moreover, with 0.1M parameters, our network demonstrates easy deployment on edge devices, e.g., phones and cameras, in real-time without GPU.

The figure below shows our approach.
<p align="center">
  <img width="800" alt="fig_method" src="https://github.com/vdquang1991/self_KD_falling_detection/blob/main/model.jpg">
</p>


## Running the code

### Requirements
- Python3
- Tensorflow (>=2.3.0)
- Numpy 
- Pillow
- Open-CV
- Scikit-learn
- ...
### Training

In this code, you can reproduce the experimental results of the falling detection task in the submitted paper.
The FDD and URFD datasets are used during the training phase.
Example training settings are for our proposed network.
Detailed hyperparameter settings are enumerated in the paper.

- Training with Self-KD
~~~
python train_tensor2.py --gpu=0 ----clip_len=16 --crop_size=224 --alpha=0.1 --use_mse=1 --lr=0.01 --drop_rate=0.5 --reg_factor=5e-4 --epochs=300 
~~~

where, 

`--clip_len` is the length of the input video clips (number of frames).

`--crop_size`is the height and width size of all frames.

`--alpha` is the loss weight for the self-distillation loss (`=lambda` in the paper).

`--use_mse=1` if using our Self-KD approach otherwise `--use_mse=0`

`--lr` is learning rate init.

`--drop_rate` is drop rate of the dropout layer.

`--reg_factor` denote weight decay.

`--epochs` is the number of epochs for training. 

### Dataset
In our work, we have used two standard datasets including FDD and URFD, to evaluate the SKD's performance compared to state-of-the-art methods.
All videos in each dataset are extracted into frames before training. 
To extract frames from video, you can utilize the `ffmpeg` command. 

For examples:

~~~
ffmpeg -i video_path.avi  "-vf", "fps=25" folder_path/%05d.jpg
~~~

To create .csv files, please see `def split_train_test_data` in `gen_tensor2.py`. The structure of the csv files is as follows:
~~~
<FOLDER>,<#START_FRAMES>,<#END_FRAMES>,<LABEL>
~~~
For examples:
~~~
Coffee_room/Videos/video_38,288,304,0
Home/Videos/video_57,16,32,0
Coffee_room/Videos/video_54,160,176,1
Coffee_room/Videos/video_15,0,16,0
Coffee_room/Videos/video_11,128,144,0
~~~



