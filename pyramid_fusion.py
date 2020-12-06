import math

import cv2
import numpy as np


def normalize_input(I):
  _min = np.min(I)
  _max = np.max(I)
  mag = _max - _min
  normalized = (I - _min) / mag
  return normalized

def build_W(I):
  sigma = 0.3
  W = np.exp(-(((I - 0.5) ** 2) / (2 * sigma * sigma)))
  return W

def build_norm_Ws(W1, W2):
  norm_W1 = np.zeros(W1.shape).astype(np.float64)
  norm_W2 = np.zeros(W1.shape).astype(np.float64)

  Y, X = W1.shape

  for y in range(Y):
    for x in range(X):
      total = W1[y, x] + W2[y, x]

      norm_W1[y, x] = W1[y, x] / total
      norm_W2[y, x] = W2[y, x] / total

  return norm_W1, norm_W2

def build_I1(V):
  alpha = 0.5

  V_clone = V[:,:].astype(np.float64)
  V_clone += 1
  I1 = np.log(alpha * V_clone.astype(np.float64))

  return I1

def build_I2(V):
  def compute_gamma():
    Y, X = V.shape
    N = Y * X
    n = 0
    for y in range(Y):
      for x in range(X):
        if V[y, x] < 50:
          n += 1
    return (N - n) / N
  
  gamma = compute_gamma()

  V_clone = V[:,:].astype(np.float64)
  I2 = 255 - (255 - V_clone) ** gamma

  return I2

def fusion(norm_I1, norm_I2, norm_W1, norm_W2):
  Y, X = norm_I1.shape

  depth = 5

  g_W1, _ = build_pyramids(norm_W1, depth)
  g_W2, _ = build_pyramids(norm_W2, depth)
  _, l_I1 = build_pyramids(norm_I1, depth)
  _, l_I2 = build_pyramids(norm_I2, depth)

  V = [np.zeros((1, 1))] * depth

  for d in range(depth):
    V[d] = np.zeros(g_W1[d].shape).astype(np.float64)
    Y, X = V[d].shape

    for y in range(Y):
      for x in range(X):
        V[d][y, x] = g_W1[d][y, x] * l_I1[d][y, x] + g_W2[d][y, x] * l_I2[d][y, x]

  for d in range(depth - 2, -1, -1):
    shape = V[d].shape
    V[d] = cv2.add(V[d], cv2.pyrUp(V[d + 1], dstsize=(shape[1], shape[0])))
  
  return V[0]

def get_shape(I, depth):
  if depth == 0:
    return I.shape 
  else:
    Y, X = get_shape(I, depth - 1) 
    return ((Y + 1) // 2, (X + 1) // 2)

def build_pyramids(I, depth=5):
  gaussian_pyramids = [I]

  cur = I
  for d in range(1, depth):
    shape = get_shape(I, d)
    new = cv2.pyrDown(cur)
    gaussian_pyramids.append(new)
    cur = new

  laplacian_pyramids = [gaussian_pyramids[depth - 1]]

  for d in range(depth - 2, -1, -1):
    shape = get_shape(I, d)
    gaussian_expanded = cv2.pyrUp(gaussian_pyramids[d + 1], dstsize=(shape[1], shape[0]))
    laplacian = cv2.subtract(gaussian_pyramids[d], gaussian_expanded)
    laplacian_pyramids.append(laplacian)

  laplacian_pyramids = list(reversed(laplacian_pyramids))

  return gaussian_pyramids, laplacian_pyramids

def enhance(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  V = hsv_img[:,:,2]
  
  print(np.min(V), np.max(V))

  I1 = build_I1(V)
  I2 = build_I2(V)

  norm_I1 = normalize_input(I1)
  norm_I2 = normalize_input(I2)

  W1 = build_W(norm_I1)
  W2 = build_W(norm_I2)

  norm_W1, norm_W2 = build_norm_Ws(W1, W2)

  new_V = fusion(norm_I1, norm_I2, norm_W1, norm_W2)
  new_V = (new_V * 255).astype(np.uint8)

  hsv_img[:,:,2] = new_V

  enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
  
  return enhanced_img