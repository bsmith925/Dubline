# Drop-in replacement for MuseTalk's musetalk/utils/preprocessing.py that needs no
# OpenMMLab stack (mmcv/mmdet/mmpose/DWPose). Landmarks come from face_alignment's
# 2D FAN network (68 iBUG points, the same indexing DWPose's keypoints[23:91]
# exposed), with its bundled S3FD detector for the fallback box. Tested recipe:
# torch 2.8 cu128 on RTX 50-series (sm_120); see setup.sh --with-musetalk.
import numpy as np
import cv2
import torch
from tqdm import tqdm
from face_alignment import FaceAlignment, LandmarksType

device = "cuda" if torch.cuda.is_available() else "cpu"
# Weights auto-download on first use (2DFAN4, s3fd) into $TORCH_HOME/hub/checkpoints.
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device, face_detector="sfd")

# Marker used by the inference scripts when no usable face box exists (kept verbatim).
coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(img_list):
    frames = []
    print("reading images...")
    for img_path in tqdm(img_list):
        frames.append(cv2.imread(img_path))
    return frames


def _detect(frame_bgr):
    """Return (landmarks[68,2] int32, detector bbox (x1,y1,x2,y2)) for the largest face, or (None, None)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    landmarks, _, boxes = fa.get_landmarks_from_image(rgb, return_bboxes=True)
    if landmarks is None or len(landmarks) == 0:
        return None, None
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    index = int(np.argmax(areas))
    points = np.asarray(landmarks[index][:, :2]).astype(np.int32)
    box = np.clip(boxes[index][:4], 0, None)
    return points, tuple(int(value) for value in box)


def _landmark_bbox(points, detector_box, upperbondrange, avg_minus, avg_plus):
    half_face_coord = points[29].copy()                      # nose-bridge midpoint (iBUG 29)
    avg_minus.append((points[30] - points[29])[1])
    avg_plus.append((points[29] - points[28])[1])
    if upperbondrange != 0:
        half_face_coord[1] = upperbondrange + half_face_coord[1]
    half_face_dist = np.max(points[:, 1]) - half_face_coord[1]
    upper_bond = max(0, half_face_coord[1] - half_face_dist)
    landmark_box = (int(np.min(points[:, 0])), int(upper_bond), int(np.max(points[:, 0])), int(np.max(points[:, 1])))
    x1, y1, x2, y2 = landmark_box
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        print("error bbox:", detector_box)
        return detector_box
    return landmark_box


def _range_text(count, avg_minus, avg_plus, upperbondrange):
    if not avg_minus:
        return f"Total frame:「{count}」 no face detected"
    return (f"Total frame:「{count}」 Manually adjust range : "
            f"[ -{int(sum(avg_minus) / len(avg_minus))}~{int(sum(avg_plus) / len(avg_plus))} ] , "
            f"the current value: {upperbondrange}")


def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    avg_minus, avg_plus = [], []
    for frame in tqdm(frames):
        points, box = _detect(frame)
        if points is None:
            continue
        _landmark_bbox(points, box, upperbondrange, avg_minus, avg_plus)
    return _range_text(len(frames), avg_minus, avg_plus, upperbondrange)


def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    coords_list = []
    avg_minus, avg_plus = [], []
    print("get key_landmark and face bounding boxes with the bbox_shift:" if upperbondrange != 0
          else "get key_landmark and face bounding boxes with the default value", upperbondrange)
    for frame in tqdm(frames):
        points, box = _detect(frame)
        if points is None:
            coords_list.append(coord_placeholder)
            continue
        coords_list.append(_landmark_bbox(points, box, upperbondrange, avg_minus, avg_plus))
    print("*" * 40 + "bbox_shift parameter adjustment" + "*" * 40)
    print(_range_text(len(frames), avg_minus, avg_plus, upperbondrange))
    print("*" * 111)
    return coords_list, frames
