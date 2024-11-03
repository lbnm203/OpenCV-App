import os
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import sys
import random
import altair as alt
# Modify the sys.path to include the correct directory
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Update the import to use relative import
from servicess.Sematic_Keypoint_Detection.draw_p import draw_points, DATATYPES, SERVICE_DIR

@st._fragment()
def display_datasets():
    st.header("1. Synthetic Shapes Datasets")
    st.divider()
    st.write(
        """
        - **Synthetic Shapes Datasets** là một tập hợp dữ liệu được thiết kế để phục vụ cho các nhiệm vụ nghiên cứu và ứng dụng
          trong lĩnh vực xử lý ảnh và thị giác máy tính. Tập dữ liệu này bao gồm nhiều hình dạng cơ bản được vẽ và sắp xếp một
          cách có cấu trúc, giúp người dùng có thể kiểm tra các mô hình nhận dạng và phát hiện hình dạng, phân loại, cũng như các
          phương pháp tăng cường dữ liệu. 
        - Tập dữ liệu có tổng cộng 4000 ảnh, được chia thành 8 loại hình và mỗi loại có 500 mẫu. Minh họa như hình bên dưới: 
        """
    )

    cols1 = st.columns(4)
    cols2 = st.columns(4)

    # Hiển thị hình ảnh ngẫu nhiên từ 4 loại hình đầu tiên
    for i in range(4):
        # Lấy danh sách các tên tệp chung trong thư mục ảnh
        common_files = [f.split('.')[0] for f in os.listdir(os.path.join(DATATYPES[i], "images"))]
        file_name = random.choice(common_files)  # Chọn ngẫu nhiên một tên tệp

        # Đọc tệp điểm và ảnh với tên đã chọn
        points = np.load(os.path.join(DATATYPES[i], "points", f"{file_name}.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i], "images", f"{file_name}.png"))
        
        caption = os.path.basename(DATATYPES[i]).replace("draw_", "")
        cols1[i].image(
            draw_points(image, points),
            use_column_width=True,
            caption=caption,
        )

    # Hiển thị hình ảnh ngẫu nhiên từ 4 loại hình còn lại
    for i in range(4):
        common_files = [f.split('.')[0] for f in os.listdir(os.path.join(DATATYPES[i + 4], "images"))]
        file_name = random.choice(common_files)

        points = np.load(os.path.join(DATATYPES[i + 4], "points", f"{file_name}.npy"))
        image = cv2.imread(os.path.join(DATATYPES[i + 4], "images", f"{file_name}.png"))
        
        caption = os.path.basename(DATATYPES[i + 4]).replace("draw_", "")
        cols2[i].image(
            draw_points(image, points),
            use_column_width=True,
            caption=caption,
        )




@st._fragment()
def display_discussion():
    st.header("5. Thảo luận")
    st.divider()
    st.write(
        """
    - Loại hình **checkerboard** và **cube**:
        - Cả hai phương pháp SIFT và ORB đều có Precision và Recall cao, cho thấy cả hai phương pháp đều hoạt động tốt trong việc
        phát hiện keypoints trên các loại hình này.
        - SIFT có lợi thế hơn ORB một chút, đặc biệt là ở checkerboard, nơi Precision của SIFT cao hơn đáng kể so với ORB.
    - Loại hình **lines**:
        - ORB có Precision cao hơn SIFT, điều này cho thấy ORB có thể phát hiện keypoints chính xác hơn trên hình dạng này.
        - Tuy nhiên, Recall của ORB thấp hơn SIFT, có nghĩa là ORB có thể bỏ sót một số keypoints tiềm năng, trong khi SIFT
        thu thập được nhiều keypoints hơn nhưng có độ chính xác thấp hơn.
    - Loại hình **star**:
        - SIFT và ORB đều đạt Precision và Recall cao, nhưng SIFT vẫn có ưu thế hơn về cả hai chỉ số.
        - Điều này chỉ ra rằng SIFT có thể phát hiện các keypoints tốt hơn trên các hình dạng có cấu trúc phức tạp như loại **star**
    - Loại hình **multiple_polygons** và **polygon**:
        - ORB có Precision cao hơn SIFT trên hai loại hình này, nhưng SIFT lại có Recall cao hơn.
    - Loại hình **stripes**:
        - SIFT có Recall cao hơn đáng kể so với ORB, trong khi Precision của ORB chỉ hơi nhỉnh hơn.
    """
    )


@st._fragment()
def display_evaluation():
    st.header("3. Đánh giá")
    st.divider()

    st.write(
        """
        - Hai độ đo **Precision** và **Recall** được sử dụng để đánh giá kết quả phát hiện keypoint của hai thuật toán **SIFT** và **ORB**
            - **Recall (độ nhạy)** đo lường khả năng của mô hình trong việc phát hiện các trường hợp dương tính thực sự.
              Nó được tính bằng tỷ lệ giữa số lượng trường hợp dương tính được phát hiện và tổng số trường hợp dương
              tính thực sự. Một **Recall** cao cho thấy mô hình có khả năng tìm ra hầu hết các trường hợp dương tính,
              nhưng không đảm bảo rằng tất cả các trường hợp dương tính được phát hiện đều chính xác.
            - **Precision (độ chính xác)** đo lường tỷ lệ giữa số lượng trường hợp dương tính được mô hình phát hiện và
              tổng số trường hợp mà mô hình dự đoán là dương tính. **Precision** cao cho thấy khi mô hình dự đoán một trường
              hợp là dương tính, xác suất để nó thực sự dương tính là cao.
        - Minh họa công thức của 2 độ đo:
        """
    )
    st.columns([1, 3, 1])[1].image(
        ("assets/pre-recall.png"),
        use_column_width=True,
        caption="Precision và Recall",
    )

    st.markdown(
        """
        - Keypoint đó được cho là dự đoán đúng nếu khoảng cách Euclidean của Keypoint đó so với khoảng cách của Keypoint thực tế không lớn hơn threshold = 5.
        - Công thức Euclidean:
        $d(true, predict) = \sqrt{(x_{true} - x_{predict}) ^ 2 + (y_{true} - y_{predict}) ^ 2}$

        """
    )


sift = cv2.SIFT_create()
orb = cv2.ORB_create()


@st._fragment()
def display_methods():
    st.header("2. Phương pháp")
    st.divider()

    st.subheader("2.1. Thuật toán SIFT")
    st.markdown(
        """
        ##### 2.1.1. Giới thiệu về thuật toán SIFT
        - **SIFT** SIFT (Scale-Invariant Feature Transform) là một thuật toán mạnh mẽ trong lĩnh vực xử lý hình ảnh,
          được phát triển bởi David Lowe vào năm 2004 trong bài báo 
                [*Distinctive Image Features from Scale-Invariant Keypoints*](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cc58efc1f17e202a9c196f9df8afd4005d16042a). Thuật toán này chủ yếu được sử dụng để phát hiện và 
          mô tả các đặc trưng quan trọng trong hình ảnh, giúp nhận diện và phân loại đối tượng một cách hiệu quả.

        Các bước trong thuật toán SIFT 

        - Phát hiện điểm đặc trưng (Keypoint Detection):
            - SIFT sử dụng Gaussian Pyramid để tạo ra các phiên bản của ảnh gốc ở nhiều mức độ mờ khác nhau. Ảnh sau khi làm mờ sẽ được giảm kích thước để tạo thành một chuỗi ảnh đa tỷ lệ (scale space).
            - Sau đó, SIFT áp dụng toán tử DoG (Difference of Gaussian) để tìm sự khác biệt giữa hai ảnh lân cận trong scale space. Các điểm cực trị trong DoG (cả tối đa và tối thiểu) là những điểm đặc trưng tiềm năng.
        
        - Định vị điểm đặc trưng (Keypoint Localization):
            - Các điểm cực trị từ bước trước sẽ được kiểm tra để loại bỏ các điểm không ổn định (như điểm nằm ở viền ảnh hoặc điểm có độ tương phản thấp).
        
        - Định hướng cho điểm đặc trưng (Orientation Assignment):
            - Mỗi điểm đặc trưng sẽ được gán một hướng (orientation) dựa trên độ dốc của ảnh tại vùng lân cận của điểm đó. Hướng này giúp điểm đặc trưng không thay đổi khi ảnh quay.
        
        - Mô tả điểm đặc trưng (Keypoint Descriptor):
            - Xung quanh mỗi điểm đặc trưng, một vector đặc trưng sẽ được tạo ra bằng cách chia vùng lân cận của điểm đó thành các ô nhỏ.
            - Trong mỗi ô, độ lớn và hướng của gradient sẽ được ghi nhận và xây dựng thành một histogram để tạo thành vector đặc trưng.
            - Vector này sau đó sẽ được chuẩn hóa để giảm độ nhạy cảm với thay đổi về ánh sáng và độ tương phản.
        
        - Ghép các đặc trưng (Keypoint Matching):
            - Các vector đặc trưng của SIFT từ hai ảnh khác nhau có thể được so sánh để tìm các cặp điểm tương đồng.
            Thông thường, phương pháp ghép dựa trên khoảng cách Euclidean giữa các vector đặc trưng. Những cặp điểm có khoảng cách gần nhau sẽ được coi là cặp điểm tương đồng.
        """
    )


    st.markdown("##### 2.1.2. Các bước chính của thuật toán SIFT:")
    # st.image(
    #     os.path.join(SERVICE_DIR, "SIFT-process.png"), use_column_width=True
    # )
    image_path = os.path.join(SERVICE_DIR, "SIFT-process.png")
    st.image(image_path, width=1000)

    st.markdown("##### 2.1.3. Minh họa ví dụ trên thuật toán SIFT:")
    cols = [st.columns(4) for _ in range(8)]

    for i in range(8):
        # Lấy danh sách các tệp chung cho cả ảnh và ground truth
        common_files = [f.split('.')[0] for f in os.listdir(os.path.join(DATATYPES[i], "images"))]
        file_name = random.choice(common_files)  # Chọn tên tệp ngẫu nhiên

        # Đọc ảnh
        image = cv2.imread(os.path.join(DATATYPES[i], "images", f"{file_name}.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện keypoints bằng SIFT
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    
        # Vẽ các keypoints SIFT lên ảnh gốc mà không vẽ ground truth
        sift_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

        # Xác định vị trí hiển thị ảnh
        caption = os.path.basename(DATATYPES[i]).replace("draw_", "")
        r, c = i // 4, i % 4
        cols[r][c].image(
            sift_image,
            use_column_width=True,
            caption=caption
        )


    st.subheader("2.2. Thuật toán ORB")
    st.markdown(
        """
        ##### 2.2.1. Giới thiệu về thuật toán ORB
        - **ORB** (Oriented FAST and Rotated BRIEF)  là một thuật toán phát hiện và mô tả đặc trưng trong hình ảnh,
          được phát triển bởi Ethan Rublee, Vincent Rabaud, Kurt Konolige và Gary R. Bradski vào năm 2011 trong bài báo 
        [*ORB: An efficient alternative to SIFT or SURF*](https://d1wqtxts1xzle7.cloudfront.net/90592905/145_s14_01-libre.pdf?1662172284=&response-content-disposition=inline%3B+filename%3DORB_An_efficient_alternative_to_SIFT_or.pdf&Expires=1729524210&Signature=dzhjTEuC-108NuiZwUIbVZStCXz5tryasM0l0sJpPx5kdMxzlIQ9ypiVK-Nrr7U4jRASqrmcG-7n0Q9nJhhEZpdOrtUd-Jw4zuBd53Z1tDUPa9BhZqImVlP3cQgAvMzsdoTsrV~yFTinoWzUKuuURdUn8jsWkqCgOzXur~sMPB4Svlihs-vGyIOL1b1hRbGLrNqwUM9KRXQAJuz2-kndG9S1zf-BReO262Qrjv7pmcgA6k4QdxajVDYqOnQDO89xUp2P0CjQwW0pwiOJ~RctdWw1fXQo2tmPPKvNsB-iXkOdKApkigZN27cAR2NH2NA39VOy~MkKHe1LefLCaSIeRw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) nhằm cải thiện tốc độ và hiệu quả so với các phương pháp trước đó như SIFT và SURF.
        **ORB** kết hợp hai kỹ thuật chính: **FAST (Features from Accelerated Segment Test)** để phát hiện các điểm
        đặc trưng, và **BRIEF (Binary Robust Invariant Scalable Keypoints)** để mô tả chúng.


        Các bước trong thuật toán ORB:

        - Phát hiện điểm đặc trưng bằng FAST

        - Tính toán hướng cho điểm đặc trưng 
            - Sau khi phát hiện điểm đặc trưng, ORB tính toán hướng của chúng bằng cách gán hướng cho mỗi điểm đặc trưng,
              ORB có thể tạo ra các đặc trưng có tính bất biến với góc xoay.

        - Tạo bộ mô tả đặc trưng bằng BRIEF:
            - ORB sử dụng bộ mô tả đặc trưng BRIEF, một bộ mô tả nhị phân giúp giảm thiểu kích thước dữ liệu và tăng tốc độ tính toán.
            - BRIEF mô tả điểm đặc trưng bằng cách so sánh cường độ của các điểm ảnh trong vùng lân cận và tạo ra một vector nhị phân.

        - Ghép các đặc trưng:
            -  Các điểm đặc trưng từ hai ảnh có thể được so khớp. Những điểm có khoảng cách nhỏ nhất sẽ được coi là các cặp điểm tương đồng.     

        """
    )

    st.markdown("##### 2.2.2. Các bước chính của thuật toán ORB:")
    st.image(
        os.path.join(SERVICE_DIR, "ORB-process.png"), width=800
    )

    st.markdown("##### 2.1.3. Minh họa ví dụ trên thuật toán ORB:")
    cols = [st.columns(4) for _ in range(8)]

    for i in range(8):
        # Lấy danh sách các tệp chung cho cả ảnh và ground truth
        common_files = [f.split('.')[0] for f in os.listdir(os.path.join(DATATYPES[i], "images"))]
        file_name = random.choice(common_files)  # Chọn tên tệp ngẫu nhiên

        # Đọc ảnh
        image = cv2.imread(os.path.join(DATATYPES[i], "images", f"{file_name}.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện keypoints bằng ORB
        keypoints = orb.detect(gray, None)
        
        # Vẽ các keypoints ORB lên ảnh gốc mà không vẽ ground truth
        orb_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)


        # Xác định vị trí hiển thị ảnh
        caption = os.path.basename(DATATYPES[i]).replace("draw_", "")
        r, c = i // 4, i % 4
        cols[r][c].image(
            orb_image,
            use_column_width=True,
            caption=caption
        )


pr_sift: np.ndarray = np.load(os.path.join(SERVICE_DIR, "pr_sift.npy"))
pr_orb: np.ndarray = np.load(os.path.join(SERVICE_DIR, "pr_orb.npy"))


@st._fragment()
def display_results():
    st.header("4. Kết quả")

    precision_sift = pr_sift[:, :, 0]
    recall_sift = pr_sift[:, :, 1]
    precision_orb = pr_orb[:, :, 0]
    recall_orb = pr_orb[:, :, 1]

    average_precision_sift = precision_sift.mean(axis=1)
    average_recall_sift = recall_sift.mean(axis=1)
    average_precision_orb = precision_orb.mean(axis=1)
    average_recall_orb = recall_orb.mean(axis=1)

    cols = st.columns(2)

    with cols[0]:
        st.subheader("Đánh giá trên độ đo Precision")
        precision_df = pd.DataFrame(
            {
                "shape_type": [
                    os.path.basename(DATATYPES[i]).split("/")[-1].replace("draw_", "")
                    for i in range(len(DATATYPES))
                ],
                "SIFT": average_precision_sift,
                "ORB": average_precision_orb,
            }
        )
        st.bar_chart(
            precision_df,
            x="shape_type",
            y_label="",
            x_label="Precision",
            horizontal=False,
        )

    with cols[1]:
        st.subheader("Đánh giá trên độ đo Recall")
        recall_df = pd.DataFrame(
            {
                
                "shape_type": [
                    os.path.basename(DATATYPES[i]).split("/")[-1].replace("draw_", "")
                    for i in range(len(DATATYPES))
                ],
                "SIFT": average_recall_sift,
                "ORB": average_recall_orb,
            }
        )
        st.bar_chart(
            recall_df,
            x="shape_type",
            y_label="",
            x_label="Recall",
            horizontal=False,

        )




def run():
#     st.set_page_config(
#     page_title="Semantic Keypoint Detection",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
    st.title("Semantic Keypoint Detection")

    display_datasets()
    display_methods()
    display_evaluation()
    display_results()
    display_discussion()
