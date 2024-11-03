import os
import cv2
import numpy as np
import streamlit as st


SERVICE_DIR = "servicess/Sematic_Keypoint_Detection"
DATASET_DIR = os.path.join(SERVICE_DIR, "synthetic_shapes_datasets")
DATATYPES = [
    os.path.join(DATASET_DIR, "draw_checkerboard"),
    os.path.join(DATASET_DIR, "draw_cube"),
    os.path.join(DATASET_DIR, "draw_ellipses"),
    os.path.join(DATASET_DIR, "draw_lines"),
    os.path.join(DATASET_DIR, "draw_multiple_polygons"),
    os.path.join(DATASET_DIR, "draw_polygon"),
    os.path.join(DATASET_DIR, "draw_star"),
    os.path.join(DATASET_DIR, "draw_stripes"),
]

sift = cv2.SIFT_create()
orb = cv2.ORB_create()


def draw_points(
    image: np.ndarray, points: np.ndarray, color=(0, 255, 0), thickness=2, radius=1
):
    for point in points:
        cv2.circle(image, (int(point[1]), int(
            point[0])), radius, color, thickness)
    return image
