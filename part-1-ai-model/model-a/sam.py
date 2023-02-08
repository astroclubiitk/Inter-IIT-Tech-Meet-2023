from PIL import Image
import numpy as np
import os
import cv2

def SAM(pred_img,org_img):
  pred_image = Image.open(pred_img)
  org_image = Image.open(org_img)
  A = np.asarray(pred_image)
  B = np.asarray(org_image)
  A = A/256
  B = B/256
  numerator = np.sum(np.multiply(A, B), axis=2)
  denominator = np.linalg.norm(A, axis=2) * np.linalg.norm(B, axis=2)
  val = np.clip(numerator / denominator, -1, 1)
  sam_angles = np.arccos(val)
  sam_angles = sam_angles * 180.0 / np.pi
  return np.mean(np.nan_to_num(sam_angles))

if __name__=="__main__":
  # img = cv2.imread(os.path.join(os.getcwd(),"data","Set5","LRbicx4","img1.png"), 1)
  # large_img = cv2.resize(img, (4000,4000), interpolation=cv2.INTER_CUBIC)
  # cv2.imwrite(os.path.join(os.getcwd(),"results","test","SRGAN_16x","cv2gen_4000.png"), large_img)
  p1 = os.path.join(os.getcwd(),"results","test","SRGAN_16x","cv2gen_4000.png")
  p2 = os.path.join(os.getcwd(),"results","test","SRGAN_16x","ohrc.png")
  print(SAM(p1,p2))