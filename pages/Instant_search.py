import cv2
import numpy as np
import streamlit as st
import os
import torch
import sys

sys.path.append('./servicess/Instance_Search')
from Superpoint import SuperPointNet

# st.set_page_config(layout='wide')

# Predefined dataset folder path
data_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../servicess/Instance_Search/storm')
)


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def load_superpoint_model():
    model = SuperPointNet()
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'superpoint_v1.pth'))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def image_query_matching(query_image, dataset_images, use_sift=True, use_orb=True, use_superpoint=False):
    # Initialize feature detectors
    if use_sift:
        sift = cv2.SIFT_create()
    if use_orb:
        orb = cv2.ORB_create()
    if use_superpoint:
        superpoint = load_superpoint_model()
    
    results = []

    # Iterate over dataset images
    for dataset_image in dataset_images:
        # Initialize lists for keypoints and descriptors
        keypoints_query, descriptors_query = [], []
        keypoints_dataset, descriptors_dataset = [], []

        # Detect features in the query image
        if use_sift:
            kp, des = sift.detectAndCompute(query_image, None)
            keypoints_query.extend(kp)
            if des is not None: descriptors_query.append(des)
        if use_orb:
            kp, des = orb.detectAndCompute(query_image, None)
            keypoints_query.extend(kp)
            if des is not None: descriptors_query.append(des)
        if use_superpoint:
            # SuperPoint inference for query image
            input_tensor = torch.from_numpy(query_image).unsqueeze(0).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                output = superpoint(input_tensor)
                kp = output['keypoints'][0].numpy()
                des = output['descriptors'][0].numpy().T
                keypoints_query.extend(kp)
                if des is not None: descriptors_query.append(des)

        # Detect features in the dataset image
        if use_sift:
            kp, des = sift.detectAndCompute(dataset_image, None)
            keypoints_dataset.extend(kp)
            if des is not None: descriptors_dataset.append(des)
        if use_orb:
            kp, des = orb.detectAndCompute(dataset_image, None)
            keypoints_dataset.extend(kp)
            if des is not None: descriptors_dataset.append(des)
        if use_superpoint:
            # SuperPoint inference for dataset image
            input_tensor = torch.from_numpy(dataset_image).unsqueeze(0).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                output = superpoint(input_tensor)
                kp = output['keypoints'][0].numpy()
                des = output['descriptors'][0].numpy().T
                keypoints_dataset.extend(kp)
                if des is not None: descriptors_dataset.append(des)

        # Convert descriptors to a unified format for matching
        descriptors_query = np.vstack([des for des in descriptors_query if des.shape[1] == descriptors_query[0].shape[1]])
        descriptors_dataset = np.vstack([des for des in descriptors_dataset if des.shape[1] == descriptors_dataset[0].shape[1]])

        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_query, descriptors_dataset)

        # Sort matches based on distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Store the results with a simple measure of matching quality (e.g., number of good matches)
        matching_quality = len(matches)
        results.append((dataset_image, matching_quality))

    # Sort results based on matching quality
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results

# Streamlit interface
def main():
    st.title("Image Query Matching System")
    st.sidebar.header("Feature Detector Options")
    use_sift = st.sidebar.checkbox("Use SIFT", value=True)
    use_orb = st.sidebar.checkbox("Use ORB", value=True)
    use_superpoint = st.sidebar.checkbox("Use SuperPoint", value=False)

    query_image_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])
    if query_image_file is not None:  # Kiểm tra nếu file đã được upload
        with st.expander("Thông tin ảnh truy vấn"):
            st.image(query_image_file, channels="BGR", width=850)
    else:
        st.warning("Vui lòng upload một file ảnh để tiếp tục.")

    with st.expander("Thông tin ảnh truy vấn"):
        st.image(query_image_file, channels="BGR", width=850)

    k = st.slider('Chọn số lượng ảnh tương đồng cần hiển thị', 1, 30, 1)
    if st.button('Tìm kiếm'):
        with st.spinner('Đang tìm kiếm'):
            if query_image_file:
                query_image = cv2.imdecode(np.frombuffer(query_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
                dataset_images = load_images_from_folder(data_folder_path)    

                if not dataset_images:
                    st.error("No images found in the specified folder.")
                else:
                    results = image_query_matching(query_image, dataset_images, use_sift=use_sift, use_orb=use_orb, use_superpoint=use_superpoint)
                    results = results[:k]  

                    with st.expander("Kết quả"):

                        for row_start in range(0, len(results), 5):
                            cols = st.columns(5)
                            for idx, (image, quality) in enumerate(results[row_start:row_start + 5]):
                                with cols[idx]:
                                    st.image(image, caption=f"Dataset Image {row_start + idx + 1} - Matching Quality: {quality}", use_column_width=True, channels="BGR")

