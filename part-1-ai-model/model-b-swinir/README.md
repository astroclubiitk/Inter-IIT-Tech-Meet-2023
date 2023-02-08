# SwinIR-PyTorch Implementation

## Overview

## Walkthrough Filesystem

- `main_train_psnr.py` : run this file to train the model from scratch or continue training of the LightWeight SR model
- `main_test_swinir.py` : run this file to generate a 16 times superscaled image from TMC2 image with the corresponding json file
- `options/swinir/train_swinir_sr_lightweight.json` : json file containing parameters to be configured while training the model from scratch or from previous checkpoint
- `model_zoo/swinir` : stores pretrained weights for generator and generator_EMA
- `dataset_sr.py` : dataset preparation and loading for training and testing
- `dataset.py` : creates batches of tensors from training or testing data using dataloader
- `requirements.txt` : environment requirements

- `train_data` : store train data here
- `superresolution/swinir_sr_lightweight_x4/models` : stores the checkpoints, weights of generator and generator_EMA and optimizer
- `superresolution/swinir_sr_lightweight_x4/logs` : stores the log files during training
- `superresolution/swinir_sr_lightweight_x4/options` : stores the json files
- `results/swinir_sr_lightweight_x4` : stores the generated images during evaluation

## Datasets

- [Google Drive](https://drive.google.com/file/d/1boCR3boKfQRaq0yukc_wm9tqSFkV0Ba-/view?usp=share_link)
- Download the zip file in above link. Place it in the train_data directory in the file system, unzip `train.zip` here and use the `final/high` directory path in the `dataroot_H` (in datasets/train and dataset/test) while training and place the `datarootL` = null. Run the instructions for training the model.
-(By specifying the dataroot_L = null the model internally does bicubic downsampling and uses 96x96 input images for training).
- [Google Drive](https://drive.google.com/file/d/1vUR8wNgacHdNdCQ_k-FnP-mdiOE9tHMA/view?usp=share_link)
- Download the `dim1x.zip` file in the above link and place it in the testsets directory and use the same path in folder_lq while testing to generate 16x upscaled image in the results directory.
- 
Note : The current working directory running the bash commands should be `swinir_implementation`.

## Weights

The pretrained weights of the generator model can be found in the link below

```https://drive.google.com/drive/folders/1mQeAeYpQtKvvvwAWPHvUMaK3taEaUKUr?usp=share_link``` and can be placed in the model_zoo/swinir directory

## How to configure the model for training

Following are the parameters which are required to be specified before training the model:-
- `gpu_ids` : stores device ids of cuda GPUs used for training, example, [0,1,2] 
- `n_channels` : specifies the number of channels in the images, example, 1 for grayscale and 3 for colored images
- `path/pretrained_G` : path for weights of pretrained generator model
- `datasets/train/dataroot_H` : path of the folder containing the train labels i.e. high resolution images
- `datasets/train/dataroot_L` : path of the folder containing the train data i.e. low resolution images
- `datasets/test/dataroot_H` : path of the folder containing the test labels i.e. high resolution images
- `datasets/test/dataroot_L` : path of the folder containing the test data i.e. low resolution images
- `train/dataloader_num_workers` : specifies the number of workers
- `train/dataloader_batch_size` : specifies the batch size
- `train/checkpoint_save` : specifies the number of iterations after which checkpoints are saved
- `train/checkpoint_print` : specifies the number of iterations after which checkpoints are printed
- `train/checkpoint_test` : specifies the number of iterations after which model is validated

Note : The parameters in the json file are set for upscaling the test images by 4 times. To get 16 times superresolution, we are cascading the model one after the other.

## How to train the data

Run the following command in the 
```python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_psnr.py --opt options/swinir/train_swinir_sr_lightweight.json  --dist False```

## How to test the model

- `model_path_name` : variable containing the weights of the generator stored during the training phase in `superresolution/swinir_sr_lightweight_x4/models` or we can use the pretrained weights from `model_zoo/swinir`
- `lr_images_path_name` : contains the test data to be passed through the generator which will be upscaled x16 times

After setting the above variables, run the following command

```python main_test_swinir.py --task lightweight_sr --scale 4 --model_path model_path_name --folder_lq lr_images_path_name --folder_gt ''```

The final generated upscaled images are stored in `results/swinir_sr_lightweight_x4`