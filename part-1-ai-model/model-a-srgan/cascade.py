import os

import cv2
import torch
from natsort import natsorted

import imgproc
import model
import srgan_config
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


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

    # Create a folder of super-resolution experiment results
    make_directory(srgan_config.stp2_sr_dir)

    # Start the verification mode of the bsrgan_model.
    g_model.eval()


    ## first forward pass

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(srgan_config.stp0_lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(srgan_config.stp0_lr_dir, file_names[index])
        sr_image_path = os.path.join(srgan_config.stp1_sr_dir, file_names[index])
        # gt_image_path = os.path.join(srgan_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}` : first itr..")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)
        # gt_tensor = imgproc.preprocess_one_image(gt_image_path, srgan_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)


    ## second forward pass

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(srgan_config.stp1_sr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(srgan_config.stp1_sr_dir, file_names[index])
        sr_image_path = os.path.join(srgan_config.stp2_sr_dir, file_names[index])
        # gt_image_path = os.path.join(srgan_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}` : second itr..")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)
        # gt_tensor = imgproc.preprocess_one_image(gt_image_path, srgan_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

if __name__ == "__main__":
    main()
