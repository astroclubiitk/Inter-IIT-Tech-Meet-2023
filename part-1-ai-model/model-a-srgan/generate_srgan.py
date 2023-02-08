import os
import numpy as np
import cv2
import torch
from natsort import natsorted
import shutil

import imgproc
import model
import srgan_config
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def crop_dim24(src: str, dst: str, image_size: int) -> None:

    print("Cropping...")
    # Crop images
    for filename in natsorted(os.listdir(src)):
        make_directory(os.path.join(dst,filename[:-4]))
        top = 0
        left = 0
        img =  cv2.imread(os.path.join(src,filename))
        img = np.asarray(img)
        for top in range(0,image_size,24):
            for left in range(0,image_size,24):
                patch_image = img[top:top + 24, left:left + 24, ...]
                newname = filename[:-4] + "_" + str(int(top/24)+1) + "_" + str(int(left/24)+1) + ".png"
                filepath = os.path.join(dst,filename[:-4],newname)
                cv2.imwrite(filepath, patch_image)
    print("Cropping done!")


def merge_dim96(src: str, dst: str, image_size: int) -> None:

    print("Merginging...")
    div_factor = int(image_size/96)

    # Merge generated images
    for dirs in natsorted(os.listdir(src)):
        folder = os.path.join(src, dirs)
        merged_image = np.zeros(shape=(image_size,image_size,3))
        for idx,filename in enumerate(natsorted(os.listdir(folder))):
            image = cv2.imread(os.path.join(folder,filename))
            image = np.asarray(image)
            # print(image.shape)
            index_r = idx % div_factor
            index_d = int(idx / div_factor)
            merged_image[index_d*96:index_d*96+96,index_r*96:index_r*96+96,...] = image
            # print(idx)
        merged_filename = dirs + ".png"
        print(merged_filename)
        cv2.imwrite(os.path.join(dst, merged_filename),merged_image)
    print("Merginging done!")


def improve(src: str) -> None:
    sharpen = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gaussian = (np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]))/256
    for filename in natsorted(os.listdir(src)):
        img = cv2.imread(os.path.join(src, filename))
        img = cv2.filter2D(img, -1, gaussian)
        img = cv2.filter2D(img, -1, sharpen)
        cv2.imwrite(os.path.join(src,filename), img)


def clean_dir(src: str) -> None:
    
    # Delete folders
    for dirs in natsorted(os.listdir(src)):
        shutil.rmtree(os.path.join(src, dirs))


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    g_model = model.__dict__[srgan_config.g_arch_name](in_channels=srgan_config.in_channels,
                                                       out_channels=srgan_config.out_channels,
                                                       channels=srgan_config.channels,
                                                       num_rcb=srgan_config.num_rcb)
    g_model = g_model.to(device=srgan_config.device)
    print(f"Build `{srgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(srgan_config.g_model_weights_path, map_location=lambda storage, loc: storage)
    g_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{srgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(srgan_config.g_model_weights_path)}` successfully.")

    # Start the verification mode of the bsrgan_model.
    g_model.eval()

    ## first split
    crop_dim24(srgan_config.stp0_lr_dir, srgan_config.dim24_dir, 240)

    ## first forward pass
    for dirs in natsorted(os.listdir(srgan_config.dim24_dir)):
        # Get a list of test image file names.
        file_names = natsorted(os.listdir(os.path.join(srgan_config.dim24_dir, dirs)))
        total_files = len(file_names)

        print(f"Processing `{os.path.abspath(os.path.join(srgan_config.dim24_dir, dirs))}` : first itr...")

        for index in range(total_files):
            lr_image_path = os.path.join(srgan_config.dim24_dir, dirs, file_names[index])
            make_directory(os.path.join(srgan_config.dim96_dir, dirs))
            sr_image_path = os.path.join(srgan_config.dim96_dir, dirs, file_names[index])

            lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)

            # Only reconstruct the Y channel image data.
            with torch.no_grad():
                sr_tensor = g_model(lr_tensor)

            # Save image
            sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(sr_image_path, sr_image)

    ## first merge
    merge_dim96(srgan_config.dim96_dir, srgan_config.stp1_sr_dir, 960)

    ## first improve
    improve(srgan_config.stp1_sr_dir)

    ## clean intermediate files
    clean_dir(srgan_config.dim24_dir)
    clean_dir(srgan_config.dim96_dir)

    ###########################################################################################

    ## second split
    crop_dim24(srgan_config.stp1_sr_dir, srgan_config.dim24_dir, 960)

    ## second forward pass
    for dirs in natsorted(os.listdir(srgan_config.dim24_dir)):
        # Get a list of test image file names.
        file_names = natsorted(os.listdir(os.path.join(srgan_config.dim24_dir, dirs)))
        total_files = len(file_names)

        print(f"Processing `{os.path.abspath(os.path.join(srgan_config.dim24_dir, dirs))}` : second itr...")

        for index in range(total_files):
            lr_image_path = os.path.join(srgan_config.dim24_dir, dirs, file_names[index])
            make_directory(os.path.join(srgan_config.dim96_dir, dirs))
            sr_image_path = os.path.join(srgan_config.dim96_dir, dirs, file_names[index])

            lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)

            # Only reconstruct the Y channel image data.
            with torch.no_grad():
                sr_tensor = g_model(lr_tensor)

            # Save image
            sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(sr_image_path, sr_image)

    ## second merge
    # make_directory(srgan_config.stp2_sr_dir)
    merge_dim96(srgan_config.dim96_dir, srgan_config.stp2_sr_dir, 3840)

    ## second improve
    improve(srgan_config.stp2_sr_dir)

    ## clean intermediate files
    clean_dir(srgan_config.dim24_dir)
    clean_dir(srgan_config.dim96_dir)

    for name in os.listdir(srgan_config.stp2_sr_dir):
        img = cv2.imread(os.path.join(srgan_config.stp2_sr_dir, name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(srgan_config.stp2_sr_dir,name), img)


if __name__ == "__main__":
    main()