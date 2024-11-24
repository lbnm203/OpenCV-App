import streamlit as st
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np

st.set_page_config(layout='wide')

sys.path.append('servicess/Object_Tracking')


def display_intro():
    st.title(" ✨ Thuật Toán SORT (Simple Online and Realtime Tracking)")
    st.divider()

    st.sidebar.checkbox("Sorting")

    st.markdown("## 1. Giới Thiệu  ")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            - SORT (Simple Online and Realtime Tracking) là một phương pháp theo dõi đối tượng đơn giản, 
            được đề xuất trong bài báo [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763). 
            Bài báo này được viết bởi **Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos và Ben Upcroft** xuất bản vào tháng 2 năm 2016
        """)

        st.markdown("""
            - Thuật toán **SORT (Simple Online and Realtime Tracking)** là một phương pháp đơn giản để theo dõi đa đối tượng 
            (Multi-Object Tracking - MOT) trong thời gian thực. Thuật toán sử dụng các thành phần cơ bản gồm bộ lọc Kalman 
            để dự đoán vị trí của đối tượng qua các khung hình và thuật toán Hungarian để thực hiện liên kết dữ liệu (data 
            association) giữa các phát hiện và đối tượng đã theo dõi.

        """)

    with col2:
        st.columns([1, 4, 1])[1].image(
            "servicess/Object_Tracking/benchmark.png",
            use_column_width=True,
            caption="Hiệu suất của thuật toán SORT so với thuật toán khác dựa trên sự so sánh giữa Speed (Tốc độ) và Accuracy (Độ chính xác) ",
        )


def display_method():

    st.markdown("## 2. Phương pháp ")

    st.markdown("""
        - Thuật toán SORT gồm có 4 bước:
            - **Bước 1:** Phát hiện đối tượng (Detection)
            - **Bước 2:** Dự đoán trạng thái đối tượng (Estimation Model) 
            - **Bước 3:** Liên kết dữ liệu (Data Association)
            - **Bước 4:** Tạo và xóa đối tượng theo dõi (Creation and Deletion of Track Identities)
    """)

    # st.image("D:\OpenCV-App\servicess\Object_Tracking\sort_alg.jpg",
    #          caption="Minh họa thuật toán SORT ")

    st.columns([1, 7.5, 1])[1].image(
        "servicess/Object_Tracking/sort_alg.jpg",
        use_column_width=True,
        caption="Hình ảnh minh họa thuật toán SORT",
    )

    st.markdown("""
        ### 2.1. Phát hiện đối tượng (Detection)
        -  SORT tận dụng các tiến bộ trong phát hiện đối tượng dựa trên Faster R-CNN để đạt được độ chính xác phát hiện cao
        Bộ phát hiện này thực hiện việc phân vùng và phân loại các đối tượng trên hình ảnh, từ đó tạo ra các hộp giới hạn cho
        mỗi đối tượng được phát hiện. Việc chọn các hộp giới hạn phụ thuộc vào xác suất dự đoán, với xác suất > 50% được xem 
        là đối tượng tiềm năng để theo dõi.
                
        ### 2.2. Dự đoán trạng thái đối tượng (Estimation Model)
        - Mô hình chuyển động của các đối tượng được ước lượng bằng một mô hình vận tốc không đổi tuyến tính độc lập với các đối 
        tượng khác và chuyển động của camera. Trạng thái của mỗi đối tượng được biểu diễn như một vector. Trạng thái của mỗi đối 
        tượng được biểu diễn như một vector:                  
                
    """)
    st.latex(r"""
        x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T
    """)

    st.markdown("""
        - Trong đó:
            - $$u$$ và $$v$$: Tọa độ của tâm bounding box (theo chiều ngang và dọc).
            - $$s$$: Tỉ lệ Diện tích tâm bounding box.
            - $$r$$: Tỉ lệ khung hình (aspect ratio) của hộp giới hạn.
            - $$\dot{u}$$ và $$\dot{v}$$: Vận tốc theo phương ngang và dọc.    
            - $$\dot{s}$$: Tốc độ thay đổi diện tích.
        
        - Bộ lọc **Kalman** được sử dụng để cập nhật và dự đoán trạng thái của đối tượng. Bộ lọc **Kalman** gồm hai bước chính:
            - Dự đoán (Prediction): Trạng thái của đối tượng tại thời điểm tiếp theo được dự đoán dựa trên mô hình chuyển động hiện tại.
            - Cập nhật (Update): Khi có phát hiện mới, bộ lọc Kalman cập nhật trạng thái bằng cách kết hợp thông tin từ phát hiện và dự đoán.
    """)

    st.markdown("""
        ### 2.3. Liên kết dữ liệu (Data Association)
        - Để liên kết các phát hiện mới với các đối tượng đã theo dõi, thuật toán dự đoán vị trí của hộp giới hạn của mỗi đối tượng trong khung 
        hình hiện tại và tính toán chi phí gán dựa trên khoảng cách Intersection over Union (IoU) giữa các hộp giới hạn dự đoán và phát hiện. 
        Thuật toán Hungarian được sử dụng để tối ưu hóa bài toán gán dữ liệu. Các phát hiện với mức độ trùng khớp nhỏ hơn ngưỡng IoU sẽ bị loại bỏ, 
        giúp cải thiện độ chính xác của việc liên kết.
    """)

    st.columns([1, 2, 1])[1].image(
        "servicess/Object_Tracking/IoU.png",
        use_column_width=True,
        caption="Hình ảnh minh họa Intersection over Union (IoU)",
    )

    st.markdown("""
    - Giá trị IoU được sử dụng để xác định mức độ phù hợp giữa các phát hiện và các đối tượng đã được theo dõi. Nếu giá trị IoU thấp hơn một ngưỡng
    nhất định ( $$IoU_{min}$$ ), phát hiện đó sẽ không được gán cho đối tượng.
    """)

    st.markdown("""
        ### 2.4. Tạo và xóa đối tượng theo dõi (Creation and Deletion of Track Identities)
        - Các danh tính theo dõi mới được tạo khi một phát hiện có độ trùng lặp thấp với tất cả các đối tượng hiện có, cho thấy sự xuất hiện của một 
        đối tượng chưa được theo dõi . Các theo dõi được chấm dứt nếu chúng không được phát hiện trong một số khung hình nhất định ( $$T_{Lost}$$ ), ngăn chặn
        sự phát triển không giới hạn của số lượng theo dõi và giảm thiểu lỗi định vị do dự đoán trong thời gian dài mà không có sửa chữa

        - Các danh tính theo dõi (track identities) được tạo ra hoặc xóa bỏ dựa trên các điều kiện:
            - Tạo đối tượng mới: Nếu có một phát hiện mới không có liên kết với bất kỳ đối tượng hiện tại nào (dựa trên giá trị IoU thấp hơn ngưỡng), 
            một bộ theo dõi mới sẽ được tạo ra.
            - Xóa đối tượng: Nếu một đối tượng không được phát hiện trong khung hình liên tiếp ( $$T_{Lost}$$ = 1), bộ theo dõi của đối tượng đó sẽ bị xóa.
    """)


def display_result():
    st.markdown(" ## 3. Kết quả minh họa")


def run():
    display_intro()
    display_method()
    display_result()
