from image_quality_assessment import ssim
import cv2
import numpy as np
import os
import srgan_config

total = 0
idx = 0

# raw = cv2.imread("F:\\ml\\SRGAN-PyTorch-main\\figure\\ohrc_2.png")
raw = cv2.imread(srgan_config.test_ohrc_path)
raw = np.asarray(raw)

dir = os.path.join(srgan_config.gen_data_dir)

for img in os.listdir(dir):
    dst = cv2.imread(os.path.join(dir,img))
    dst = np.asarray(dst)
    total += ssim(raw, dst, 10, True)
    idx +=1
    print(idx)

mean_ssim = total/idx
print(mean_ssim)