import cv2
import mimetypes
import numpy as np

from numpy.linalg import norm
from skimage import transform as trans


def is_video(path: str) -> bool:
    return "video" in mimetypes.guess_type(path)[0] if mimetypes.guess_type(path)[0] else False


def is_image(path: str) -> bool:
    return "image" in mimetypes.guess_type(path)[0] if mimetypes.guess_type(path)[0] else False


def cosine_similarity(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim


def estimate_norm(landmarks, image_size = 512, zoom_out_factor = 0.8):
    # Original code from https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
    # Modified by adding zoom_out_factor and removing support for image_size % 112

    arcface_dst = np.array([
        [38.2946, 51.6963], 
        [73.5318, 51.5014], 
        [56.0252, 71.7366],
        [41.5493, 92.3655], 
        [70.7299, 92.2041]
        ], dtype=np.float32)
    
    assert landmarks.shape == (5, 2)
    assert image_size % 128 == 0

    ratio = float(image_size) / 128.0
    diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x

    dst_center = np.mean(dst, axis=0)
    dst = (dst - dst_center) * zoom_out_factor + dst_center

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    return tform.params[0:2, :]


def norm_crop(image, landmarks, image_size=512):
    return cv2.warpAffine(
        src=image, 
        M=estimate_norm(landmarks, image_size), 
        dsize=(image_size, image_size), 
        borderValue=0.0)