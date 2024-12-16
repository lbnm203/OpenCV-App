import streamlit as st
import cv2
import numpy as np
from PIL import Image
from PIL import Image, ImageOps
from io import BytesIO

def flip_image(image, mode):
    if mode == "Horizontal":
        return cv2.flip(image, 1)
    elif mode == "Vertical":
        return cv2.flip(image, 0)
    elif mode == "Both":
        return cv2.flip(image, -1)
    return image

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def change_color_space(image, space):
    if space == "Gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif space == "Lab":
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image

def translate_image(image, x, y):
    h, w = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, translation_matrix, (w, h))

def crop_image(image, x_start, y_start, width, height):
    return image[y_start:y_start + height, x_start:x_start + width]


def main():
    st.title(" ✨ Image Processing Application")
    st.markdown("- Ứng dụng này sẽ giúp thực hiện các kỹ thuật xử lý ảnh cơ bản như: Flip, Rotation, Colorspace, Translation, Cropping.")
# Flip
    st.subheader("1. Flip")
    st.divider()
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        flip_mode = st.selectbox("Flip Mode", ["None", "Horizontal", "Vertical", "Both"])
        if flip_mode != "None":
            image = Image.open(uploaded_file)
            image = np.array(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            with col2:
                image = flip_image(image, flip_mode)
                st.image(image, caption=f"Flipped Image ({flip_mode})", use_column_width=True)

    # Rotation
    st.subheader("2. Rotation")
    st.divider()
    rot_uploader = st.file_uploader("Upload ảnh để tiến hành xoay", type=["jpg", "png", "jpeg"])
    if rot_uploader is not None:
        angle = st.slider("Rotation Angle", -180, 180, 0)
        image = Image.open(rot_uploader)
        image = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            if angle != 0:
                image = rotate_image(image, angle)
                st.image(image, caption=f"Rotated Image ({angle}°)", use_column_width=True)

    # Color Space
    st.subheader("3. Color Space Conversion")
    st.divider()
    col_uploader = st.file_uploader("Upload ảnh để chuyển màu", type=["jpg", "png", "jpeg"])
    if col_uploader is not None:
        color_space = st.selectbox("Color Space", ["Gray", "HSV", "Lab"])
        image = Image.open(col_uploader)
        image = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            if color_space:
                image = change_color_space(image, color_space)
                st.image(image, caption=f"Color Space: {color_space}", use_column_width=True)

    # Translation
    st.subheader("4. Translation")
    st.divider()
    trans_uploader = st.file_uploader("Upload ảnh để dịch chuyển ảnh", type=["jpg", "png", "jpeg"])
    if trans_uploader is not None:
        x_translation = st.slider("Translate X (pixels)", -100, 100, 0)
        y_translation = st.slider("Translate Y (pixels)", -100, 100, 0)
        image = Image.open(trans_uploader)
        image = np.array(image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            if x_translation != 0 or y_translation != 0:
                image = translate_image(image, x_translation, y_translation)
                st.image(image, caption="Translated Image", use_column_width=True)


    # Cropping
    st.subheader("5. Cropping")
    st.divider()
    crop_uploader = st.file_uploader("Upload ảnh để cắt ảnh", type=["jpg", "png", "jpeg"])
    if crop_uploader is not None:
        image = Image.open(crop_uploader)
        image = np.array(image)
        x_start = st.slider("X Start", min_value=0, max_value=image.shape[1] - 1, value=0)
        y_start = st.slider("Y Start", min_value=0, max_value=image.shape[0] - 1, value=0)
        crop_width = st.slider("Crop Width", min_value=1, max_value=image.shape[1] - x_start, value=image.shape[1])
        crop_height = st.slider("Crop Height", min_value=1, max_value=image.shape[0] - y_start, value=image.shape[0])
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        # if st.button("Crop"):
        with col2:
            image = crop_image(image, int(x_start), int(y_start), int(crop_width), int(crop_height))
            st.image(image, caption="Cropped Image", use_column_width=True)

