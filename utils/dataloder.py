import os
import numpy as np
from PIL import Image

import utils.augmentations as augmentations
def get_files(path):
    ret = []
    path_rainy = path + "/small/rain"
    path_gt = path + "/small/norain"

    for root, dirs, files in os.walk(path_rainy):
        files.sort()    
        
        for name in files:
            if name.split('.')[1] != 'png':
                continue
            file_rainy = path_rainy + "/" + name
            file_gt = path_gt + "/" + name
            ret.append([file_rainy, file_gt])
    return ret

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

  Returns:
    mixed: Augmented and mixed image.
  """
  ws = np.float32(
      np.random.dirichlet([alpha] * width))
  m = np.float32(np.random.beta(alpha, alpha))

  mix = np.zeros_like(image)
  for i in range(width):
    image_aug = image.copy()
    depth = depth if depth > 0 else np.random.randint(2, 4)
    for _ in range(depth):
      op = np.random.choice(augmentations.augmentations)
      #print(op)
      image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * normalize(image_aug)
    
  max_ws = max(ws)
  rate = 1.0 / max_ws
  #print(rate)
  
  
  #mixed = (random.randint(5000, 9000)/10000) * normalize(image) + (random.randint((int)(rate*3000), (int)(rate*10000))/10000) * mix
  mixed = max((1 - m), 0.7) * normalize(image) + max(m, rate*0.5) * mix
  #mixed = (1 - m) * normalize(image) + m * mix
  return mixed

def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  '''
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)
  '''
  return image

def apply_op(image, op, severity):
  image = np.clip(image * 255., 0, 255).astype(np.uint8)
  pil_img = Image.fromarray(image)  # Convert to PIL.Image
  pil_img = op(pil_img, severity)
  return np.asarray(pil_img) / 255.
