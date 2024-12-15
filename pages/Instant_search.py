import cv2
import numpy as np
import streamlit as st
import os
import torch
import sys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('./servicess/Instance_Search')
from Superpoint import SuperPointNet

# st.set_page_config(layout='wide')

# Predefined dataset folder path
data_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../servicess/Instance_Search/storm')
)

# Number of clusters for BOVW
n_clusters = 50  # Adjust the number of clusters based on the dataset size

@st.cache_resource
def load_superpoint_model():
    model = SuperPointNet()
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'superpoint_v1.pth'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def visualize_features(image, keypoints, title):
    # Vẽ keypoints lên ảnh
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))
    st.image(img_with_kp, caption=title, channels="BGR", use_column_width=True)

@st.cache_resource
def extract_features(images, use_sift=True, use_orb=True, visualize=True):
    # Initialize feature detectors
    sift = cv2.SIFT_create() if use_sift else None
    orb = cv2.ORB_create() if use_orb else None

    sift_descriptors, orb_descriptors = [], []
    
    # Extract features from all images
    for img in images:
        if visualize and len(images) == 1:  # Chỉ show query image nếu visualize=True
            col1, col2 = st.columns(2)
            with col1:
                if use_sift:
                    kp_sift, des = sift.detectAndCompute(img, None)
                    visualize_features(img, kp_sift, "SIFT Features")
                    if des is not None:
                        sift_descriptors.append(des)
            with col2:
                if use_orb:
                    kp_orb, des = orb.detectAndCompute(img, None)
                    visualize_features(img, kp_orb, "ORB Features")
                    if des is not None:
                        orb_descriptors.append(des)
        else:  # Không cần trực quan hóa
            if use_sift:
                _, des = sift.detectAndCompute(img, None)
                if des is not None:
                    sift_descriptors.append(des)
            if use_orb:
                _, des = orb.detectAndCompute(img, None)
                if des is not None:
                    orb_descriptors.append(des)
                    
    return sift_descriptors, orb_descriptors
        

def create_bovw_dictionary(descriptors_list, n_clusters=50):
    if len(descriptors_list) == 0:
        return None

    # Stack all descriptors together for clustering
    all_descriptors = np.vstack(descriptors_list)

    # Perform KMeans clustering to create the visual words
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    return kmeans

def compute_bovw_histogram(image_descriptors, kmeans):
    if len(image_descriptors) == 0:
        return np.zeros(kmeans.n_clusters)

    # Predict the cluster for each descriptor in the image
    words = kmeans.predict(image_descriptors)

    # Create a histogram of the visual words
    histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))

    # Normalize the histogram to have unit length
    histogram = histogram / np.linalg.norm(histogram)

    return histogram

def compute_combined_histogram(sift_descriptors, orb_descriptors, kmeans_sift, kmeans_orb):
    # Compute histogram for each type of descriptor
    sift_histogram = compute_bovw_histogram(sift_descriptors, kmeans_sift) if kmeans_sift else np.array([])
    orb_histogram = compute_bovw_histogram(orb_descriptors, kmeans_orb) if kmeans_orb else np.array([])

    # Concatenate histograms
    combined_histogram = np.concatenate([sift_histogram, orb_histogram])

    return combined_histogram

# Show image in folder (2x4)
@st.cache_data()
def show_dataset_images():
    ct = st.container(border=True)
    with ct:
        st.subheader("1. Tập dữ liệu")
        st.markdown("Tập dữ liệu dùng để truy vấn gồm **30 hình ảnh** về hư hại do bão")
        dataset_images = load_images_from_folder(data_folder_path)
        
        if not dataset_images:
            ct.error("No images found in the dataset folder.")
            return
            
        # Display only the first 8 images in a 2x4 grid
        max_images = 8  # Maximum 8 images to display
        displayed_images = dataset_images[:max_images]  # Select the first 8 images

        rows = 2
        cols_per_row = 4

        for i in range(rows):  # Loop for 2 rows
            cols = st.columns(cols_per_row)  # Create 4 columns per row
            for j in range(cols_per_row):  # Loop for 4 columns
                index = i * cols_per_row + j  # Calculate the image index
                if index < len(displayed_images):
                    with cols[j]:
                        st.image(displayed_images[index],
                                caption=f"Image {index + 1}",
                                use_column_width=True,
                                channels="BGR")


# Streamlit interface
def main():
    st.title(" ✨ Instance Search")
    show_dataset_images()
    st.sidebar.header("Feature Detector Options")
    use_sift = st.sidebar.checkbox("Use SIFT", value=True)
    use_orb = st.sidebar.checkbox("Use ORB", value=True)

    query_image_file = st.sidebar.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])
    
    if query_image_file:
        query_image = cv2.imdecode(np.frombuffer(query_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        

        # Tab cho visualization
        # feature_tab, histogram_tab, results_tab = st.tabs(["Feature Detection", "BoVW Histograms", "Search Results"])

        k = st.sidebar.slider('Chọn số lượng ảnh tương đồng cần hiển thị', 1, 30, 1)
        
        if st.sidebar.button('Tìm kiếm'):
            # with st.spinner('Đang xử lý...'):
            # Load dataset images
            dataset_images = load_images_from_folder(data_folder_path)

            if not dataset_images:
                st.error("No images found in the specified folder.")
            else:
                # with feature_tab:
                    # st.subheader("2. Feature Detection")
                    # Extract và visualize features
                ct = st.container(border=True)
                with ct:
                    st.subheader("2. Query Image")
                    st.image(query_image, channels="BGR", width=400)
                    query_sift_descriptors, query_orb_descriptors = extract_features(
                        [query_image], use_sift=use_sift, use_orb=use_orb
                    )
                                
                # with histogram_tab:
                    # st.subheader("3. Bag of Visual Words Histograms")
                    # Extract features từ dataset
                dataset_sift_descriptors, dataset_orb_descriptors = extract_features(
                    dataset_images, use_sift=use_sift, use_orb=use_orb
                )

                # Create BOVW dictionaries
                kmeans_sift = create_bovw_dictionary(
                    dataset_sift_descriptors + query_sift_descriptors, 
                    n_clusters=n_clusters
                ) if use_sift else None
                
                kmeans_orb = create_bovw_dictionary(
                    dataset_orb_descriptors + query_orb_descriptors, 
                    n_clusters=n_clusters
                ) if use_orb else None

                # Compute và visualize histograms
                query_histogram = compute_combined_histogram(
                    query_sift_descriptors[0] if use_sift else np.array([]),
                    query_orb_descriptors[0] if use_orb else np.array([]),
                    kmeans_sift, kmeans_orb
                )
                cont = st.container(border=True)
                with cont:
                    # with results_tab:
                    st.subheader("3. Search Results")
                    # Compute histograms cho dataset
                    dataset_histograms = []
                    for i in range(len(dataset_images)):
                        dataset_histogram = compute_combined_histogram(
                            dataset_sift_descriptors[i] if use_sift else np.array([]),
                            dataset_orb_descriptors[i] if use_orb else np.array([]),
                            kmeans_sift, kmeans_orb
                        )
                        dataset_histograms.append(dataset_histogram)

                    # Tính similarity và hiển thị kết quả
                    similarities = []
                    for idx, histogram in enumerate(dataset_histograms):
                        similarity = cosine_similarity([query_histogram], [histogram])[0][0]
                        similarities.append((dataset_images[idx], similarity))

                    results = sorted(similarities, key=lambda x: x[1], reverse=True)
                    results = results[:k]

                    for row_start in range(0, len(results), 5):
                        cols = st.columns(5)
                        for idx, (image, similarity) in enumerate(results[row_start:row_start + 5]):
                            if idx < len(cols):
                                with cols[idx]:
                                    st.image(image, 
                                            caption=f"Similar {row_start + idx + 1}\nScore: {similarity:.3f}", 
                                            use_column_width=True, 
                                            channels="BGR")

# def extract_features(images, use_sift=True, use_orb=True):
#     # Initialize feature detectors
#     sift = cv2.SIFT_create() if use_sift else None
#     orb = cv2.ORB_create() if use_orb else None
#     # superpoint = load_superpoint_model() if use_superpoint else None

#     # sift_descriptors, orb_descriptors, superpoint_descriptors = [], [], []
#     sift_descriptors, orb_descriptors = [], []

#     # Extract features from all images
#     for img in images:
#         if use_sift:
#             _, des = sift.detectAndCompute(img, None)
#             if des is not None:
#                 sift_descriptors.append(des)
#         if use_orb:
#             _, des = orb.detectAndCompute(img, None)
#             if des is not None:
#                 orb_descriptors.append(des)

#     return sift_descriptors, orb_descriptors

# def create_bovw_dictionary(descriptors_list, n_clusters=50):
#     if len(descriptors_list) == 0:
#         return None

#     # Stack all descriptors together for clustering
#     all_descriptors = np.vstack(descriptors_list)

#     # Perform KMeans clustering to create the visual words
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(all_descriptors)

#     return kmeans

# def compute_bovw_histogram(image_descriptors, kmeans):
#     if len(image_descriptors) == 0:
#         return np.zeros(kmeans.n_clusters)

#     # Predict the cluster for each descriptor in the image
#     words = kmeans.predict(image_descriptors)

#     # Create a histogram of the visual words
#     histogram, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))

#     # Normalize the histogram to have unit length
#     histogram = histogram / np.linalg.norm(histogram)

#     return histogram

# def compute_combined_histogram(sift_descriptors, orb_descriptors, kmeans_sift, kmeans_orb):
#     # Compute histogram for each type of descriptor
#     sift_histogram = compute_bovw_histogram(sift_descriptors, kmeans_sift) if kmeans_sift else np.array([])
#     orb_histogram = compute_bovw_histogram(orb_descriptors, kmeans_orb) if kmeans_orb else np.array([])

#     # Concatenate histograms to create a combined feature representation
#     combined_histogram = np.concatenate([sift_histogram, orb_histogram])

#     return combined_histogram
# # Streamlit interface
# def main():
#     st.title(" ✨ Instance Search")
#     st.sidebar.header("Feature Detector Options")
#     use_sift = st.sidebar.checkbox("Use SIFT", value=True)
#     use_orb = st.sidebar.checkbox("Use ORB", value=True)
#     # use_superpoint = st.sidebar.checkbox("Use SuperPoint", value=False)

#     query_image_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])
#     if query_image_file:
#         query_image = cv2.imdecode(np.frombuffer(query_image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         with st.expander("Thông tin ảnh truy vấn"):
#             st.image(query_image, channels="BGR", width=850)

#     k = st.slider('Chọn số lượng ảnh tương đồng cần hiển thị', 1, 30, 1)
#     if st.button('Tìm kiếm'):
#         with st.spinner('Đang tìm kiếm'):
#             if query_image_file:
#                 # Load dataset images
#                 dataset_images = load_images_from_folder(data_folder_path)

#                 if not dataset_images:
#                     st.error("No images found in the specified folder.")
#                 else:
#                     query_sift_descriptors, query_orb_descriptors = extract_features(
#                         [query_image], use_sift=True, use_orb=True
#                     )
#                     dataset_sift_descriptors, dataset_orb_descriptors = extract_features(
#                         dataset_images, use_sift=True, use_orb=True
#                     )

#                     # Create BOVW dictionaries for each type of feature
#                     kmeans_sift = create_bovw_dictionary(dataset_sift_descriptors + query_sift_descriptors, n_clusters=n_clusters) if use_sift else None
#                     kmeans_orb = create_bovw_dictionary(dataset_orb_descriptors + query_orb_descriptors, n_clusters=n_clusters) if use_orb else None

#                     query_histogram = compute_combined_histogram(
#                         query_sift_descriptors[0] if use_sift else np.array([]),
#                         query_orb_descriptors[0] if use_orb else np.array([]),
#                         kmeans_sift, kmeans_orb
#                     )

#                     dataset_histograms = []

#                     for i in range(len(dataset_images)):
#                         dataset_histogram = compute_combined_histogram(
#                             dataset_sift_descriptors[i] if use_sift else np.array([]),
#                             dataset_orb_descriptors[i] if use_orb else np.array([]),
#                             kmeans_sift, kmeans_orb
#                         )
#                         dataset_histograms.append(dataset_histogram)

#                     # Compute cosine similarity between query and dataset images
#                     similarities = []
#                     for idx, histogram in enumerate(dataset_histograms):
#                         similarity = cosine_similarity([query_histogram], [histogram])[0][0]
#                         similarities.append((dataset_images[idx], similarity))

#                     # Sort results based on similarity
#                     results = sorted(similarities, key=lambda x: x[1], reverse=True)
#                     results = results[:k]

#                     with st.expander("Kết quả"):
#                         for row_start in range(0, len(results), 5):
#                             cols = st.columns(5)
#                             for idx, (image, similarity) in enumerate(results[row_start:row_start + 5]):
#                                 if idx < len(cols):
#                                     with cols[idx]:
#                                         st.image(image, caption=f"Dataset Image {row_start + idx + 1} - Similarity: {similarity:.3f}", use_column_width=True, channels="BGR")
