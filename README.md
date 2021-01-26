# Deep Visual Odometry
Training and validation of Visual Odometry models.
Implementation is based on ![Huangying's paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhan_Unsupervised_Learning_of_CVPR_2018_paper.pdf) with a few changes.

## Running with wandb
Create a new project at https://app.wandb.ai/

```
WANDB_NAME='testing' CUDA_VISIBLE_DEVICES=2 python3 train.py config_kitti.yaml
```

Here:
* `WANDB_NAME` - Experiment name
* `CUDA_VISIBLE_DEVICES` - ID of a GPU used for training (starting with 0)
To see which GPUs are free: `nvidia-smi`
* `config_kitti.yaml` - Parameters of the experiment and path to the dataset.
