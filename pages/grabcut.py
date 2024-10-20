# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import streamlit as st
import tempfile
import os

from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
# from rembg import remove
from streamlit_drawable_canvas import st_canvas

# st.set_page_config(layout='wide')


def run():
    print(__doc__)
    image_upload = st.sidebar.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"])
    custom_tool_mapping = {
        "Rect": "rect",
        "Sure Foreground": "freedraw",
        "Sure Background": "freedraw"
    }

    if 'prev_image' not in st.session_state:
        st.session_state.prev_image = None

    if image_upload is not None:
        image_ul = np.array(Image.open(image_upload))
        masks = np.zeros(image_ul.shape[:2], np.uint8)
        if "masks_or" not in st.session_state:
            st.session_state.masks_or = None

        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None

        if st.session_state.prev_image is None:
            st.session_state.prev_image = image_ul
        else:
            if st.session_state.prev_image.shape != image_ul.shape:
                st.session_state.prev_image = image_ul
                st.session_state.masks_or = None
                st.session_state.masks_state = False
        choosen_tool = st.sidebar.radio(
            "Chọn chế độ vẽ:", list(custom_tool_mapping.keys()))
        drawing_mode = custom_tool_mapping[choosen_tool]
        # freedraw: foreground, line:background
        stroke_color = "green"
        if choosen_tool == "Sure Foreground":
            stroke_color = "white"
        elif drawing_mode == "rect":
            stroke_color = "green"
        else:
            stroke_color = "black"
        # stroke_width = st.sidebar.slider("Stroke width: ", 1, 3, 3)
        # realtime_update = st.sidebar.checkbox("Update in realtime", True)

        if not os.path.exists('images'):
            os.makedirs('images')
        img = Image.open(image_upload)
        # image.save('images/' + image_upload.name)

        # Tạo thành phần canvas
        # img = Image.open('images/' + image_upload.name)
        max_size = 480
        w = min(img.width, max_size)
        h = w * img.height // img.width
        # print(img.width, img.height, w, h)

        c1, c2 = st.columns([1, 0.7])
        with c1:
            st.markdown(
                '   <p style="text-indent: 200px;"> <span style = "color:orange; font-size:25px;"> Ảnh gốc</span>', unsafe_allow_html=True)
            canvas_result = st_canvas(
                # fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                fill_color="rgba(255, 255, 255, 0.3)",
                stroke_width=3,
                stroke_color=stroke_color,
                background_image=img,
                update_streamlit=True,
                width=w,
                height=h,
                drawing_mode=drawing_mode,
                key=image_upload.name,
            )
            # c1.write('Ảnh gốc')

        # canvas_result.json_data chứa thông tin các hình vẽ trên canvas

        ori_image = np.array(img)
        ori_image = cv.cvtColor(ori_image, cv.COLOR_RGBA2BGR)

        scale = ori_image.shape[1] / w

        # Tạo mask để vẽ foreground
        mask_img_fore = Image.new(
            "L", (image_ul.shape[1], image_ul.shape[0]), 0)
        draw_fore = ImageDraw.Draw(mask_img_fore)
        # Tạo mask để vẽ background
        mask_img_back = Image.new(
            "L", (image_ul.shape[1], image_ul.shape[0]), 0)
        draw_back = ImageDraw.Draw(mask_img_back)

        ok = False
        if canvas_result is not None and canvas_result.json_data is not None:
            list_rect = []
            for obj in canvas_result.json_data["objects"]:
                # Tọa độ x, y trái trên
                x = obj["left"] * scale
                y = obj["top"] * scale

                # Chiều dài, chiều rộng
                width = obj["width"] * scale
                height = obj["height"] * scale
                min_x = int(x)
                min_y = int(y)

                rect = (min_x, min_y, int(width), int(height))
                print(rect)
                list_rect.append(rect)
                if obj["type"] == "path":
                    ok = 1
                    path_points = obj["path"]
                    color = obj["stroke"]
                    points = [(int(point[1] * scale), int(point[2] * scale))
                              for point in path_points if len(point) > 2]

                    # Vẽ đường trên mặt nạ
                    if color == "white":
                        draw_fore.line(points, fill=255, width=5)
                    else:
                        draw_back.line(points, fill=255, width=5)

            mask_fg = np.array(mask_img_fore)
            if st.session_state.masks_or is None:
                st.session_state.masks_or = masks
            st.session_state.masks_or[mask_fg > 0] = cv.GC_FGD

            mask_bg = np.array(mask_img_back)
            if st.session_state.masks_or is None:
                st.session_state.masks_or = masks
            st.session_state.masks_or[mask_bg > 0] = cv.GC_BGD

            if "masks_state" not in st.session_state:
                st.session_state.masks_state = False

            if mask_fg.sum() > 0 or mask_bg.sum() > 0:
                st.session_state.masks_state = True

            rect = None
            if len(list_rect) > 0:
                rect = list_rect[-1]
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Áp dụng grapCut
            rect_or_mask = cv.GC_INIT_WITH_RECT
            if ok:
                rect_or_mask = cv.GC_INIT_WITH_MASK
            if "image_state" not in st.session_state:
                st.session_state.image_state = False

            if "submit_clicked" not in st.session_state:
                st.session_state.submit_clicked = False

            st.session_state.submit_clicked = False

            if st.button("Submit"):
                st.session_state.submit_clicked = True
                st.session_state.image_state = True

            if st.session_state.submit_clicked:
                if st.session_state.masks_or is not None:
                    masks = st.session_state.masks_or
                st.session_state.masks_or, _, _ = cv.grabCut(
                    ori_image, st.session_state.masks_or, rect, bgd_model, fgd_model, 5, rect_or_mask)
                # Sửa đổi mask để các pixel được gán nhãn là foreground là 1, còn lại là 0
                mask2 = np.where((st.session_state.masks_or == cv.GC_FGD) | (
                    st.session_state.masks_or == cv.GC_PR_FGD), 1, 0).astype('uint8')
                # Áp masks vào ảnh gốc
                grabcut_result = image_ul * mask2[:, :, np.newaxis]
                st.session_state.processed_image = grabcut_result
            if st.session_state.processed_image is not None and st.session_state.processed_image.shape == image_ul.shape:
                c2.markdown(
                    '<p style="text-align: center; color: orange; font-size: 25px;">Kết quả</p>', unsafe_allow_html=True)
                c2.image(st.session_state.processed_image)
                # c2.write('Kết quả')


def use():
    st.write("# ✨ Ứng dụng GrabCut")

    st.divider()

    st.markdown("""
        ## Hướng dẫn cách dùng
        - Tải ảnh muốn tách nền ở thanh bên trái
        - Chọn chế độ vẽ bên thanh trái
        - Giữ chuột phải vào ảnh để vẽ hình chữ nhật quanh đối tượng cần tách nền
            - Lưu ý: vẽ hình chữ nhật đầu tiên và chỉ vẽ 1 lần. 
        - Để chỉnh sửa Foreground và Background, chọn chế độ ở thanh bên trái
        - Để hoàn tác các thao tác vừa thực hiện, click vào biểu tượng ↩️↪️ dưới ảnh
        - Để xóa thao tác, click vào biểu tượng thùng rác 🗑️ dưới ảnh 
        - Ấn vào biểu tượng "Submit" để tiến hành chạy

    """)
