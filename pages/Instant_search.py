import os
import streamlit as st
import numpy as np
import cv2
import joblib
from scipy.cluster.vq import vq
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import sys

sys.path.append('servicess')
# Alternative: Use script location as reference
# model_directory = os.path.join(os.path.dirname(__file__))
model_directory = "servicess/Instance_Search"
# Đường dẫn đến thư mục chứa ảnh thử nghiệm
test_directory = os.path.join(model_directory, "test")

# Tải các mô hình và dữ liệu đã lưu
codebook_path = os.path.join(
    model_directory, "bovw_codebook.joblib")
frequency_vectors_path = os.path.join(
    model_directory, "frequency_vectors.joblib")
image_paths_path = os.path.join(
    model_directory, "image_paths.joblib")

codebook = joblib.load(codebook_path)
frequency_vectors = joblib.load(frequency_vectors_path)
image_paths = joblib.load(image_paths_path)

# Thiết lập SIFT cho việc trích xuất đặc trưng
sift = cv2.SIFT_create()
k = codebook.shape[0]  # Số lượng visual words


def extract_bovw_vector(image, codebook, k):
    # Tiền xử lý và trích xuất đặc trưng SIFT từ ảnh đầu vào
    img = np.array(image)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_resized = cv2.resize(
            img, (200, int(200 * img.shape[0] / img.shape[1])))
        img_smoothed = cv2.GaussianBlur(img_resized, (5, 5), 0)
        img_gray = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:  # Ảnh đã ở dạng grayscale
        img_gray = img
    else:
        st.error("Định dạng ảnh không hợp lệ.")
        return None

    # Trích xuất đặc trưng SIFT
    _, descriptors = sift.detectAndCompute(img_gray, None)
    if descriptors is None:
        return None

    # Mã hóa các đặc trưng thành vector BoVW
    visual_words, _ = vq(descriptors, codebook)
    bovw_vector = np.zeros(k)
    for word in visual_words:
        bovw_vector[word] += 1

    return bovw_vector


def find_similar_images(query_vector, frequency_vectors, image_paths, top_n=5):
    # Tính độ tương đồng cosine giữa vector truy vấn và tất cả vector BoVW
    similarities = cosine_similarity([query_vector], frequency_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_images = [(image_paths[i], similarities[i]) for i in top_indices]
    return top_images


def run():
    # Thiết lập giao diện Streamlit
    st.title("Instance Search")

    uploaded_file = st.file_uploader(
        "Tải lên một ảnh", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Ảnh đã tải lên",
                 use_column_width=True, width=800)

        # Trích xuất vector BoVW cho ảnh truy vấn
        top_n = st.slider("Số lượng ảnh tương đồng cần hiển thị", 1, 100, 5)
        if st.button('Tìm kiếm'):
            query_vector = extract_bovw_vector(query_image, codebook, k)
            if query_vector is not None:
                # Tìm ảnh tương tự
                similar_images = find_similar_images(
                    query_vector, frequency_vectors, image_paths, top_n=top_n)

                st.write("{top_n} Ảnh Tương Tự Nhất:")
                for img_path, similarity in similar_images:
                    # Construct full path
                    full_image_path = os.path.join(
                        test_directory, img_path)
                    if os.path.exists(full_image_path):
                        st.write(f"Độ tương đồng: {similarity:.2f}")
                        similar_image = Image.open(full_image_path)
                        st.image(similar_image, caption=full_image_path,
                                 use_column_width=True)
                    else:
                        st.warning(f"Ảnh không tồn tại: {full_image_path}")
            else:
                st.error("Không tìm thấy đặc trưng trong ảnh tải lên.")
