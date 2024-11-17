import streamlit as st
import cv2
import tempfile
import pandas as pd
import requests

def main():
    # Tạo giao diện ứng dụng Streamlit
    st.title(" ✨ Boosting Object Tracking")

    st.divider()

    # Câu trả lời
    st.markdown("### 1. Thuật toán Boosting Tracker")

    st.markdown("**Định nghĩa và Lịch sử:**")
    st.markdown("- Thuật toán Boosting Tracker được giới thiệu trong bài báo khoa học **Real-Time Tracking via On-line Boosting** bởi các tác giả **Helmut Grabner, Michael Grabner, và Horst Bischof** được xuất bản vào tháng 1 năm 2006.")
    st.markdown("- **Boosting Tracker** là một kỹ thuật theo dõi đối tượng trong thời gian thực sử dụng thuật toán AdaBoost trực tuyến để lựa chọn đặc trưng và phân loại đối tượng mục tiêu từ nền. Thuật toán này xem bài toán theo dõi như một bài toán phân loại nhị phân, trong đó đối tượng được phân biệt với nền bởi một bộ phân loại.")

    st.markdown("### 2. Các bước trong thuật toán Boosting Tracker")
    st.markdown("1. **Khởi tạo Tracker:** Lựa chọn đối tượng cần theo dõi từ khung hình đầu tiên của video. Thiết lập một bộ phân loại với AdaBoost để phân biệt giữa đối tượng và nền.")
    st.markdown("2. **Huấn luyện Bộ Phân Loại:** Sử dụng các bộ phân loại yếu (như decision stumps) để huấn luyện một bộ phân loại mạnh trên các đặc trưng của đối tượng.")
    st.markdown("3. **Cập nhật Bộ Phân Loại:** Mỗi khung hình tiếp theo, bộ phân loại sẽ được cập nhật để thích ứng với sự thay đổi của vật thể.")
    st.markdown("4. **Xác định Vị Trí Đối Tượng:** Bộ phân loại sau đó được sử dụng để xác định vị trí đối tượng trong khung hình tiếp theo, bằng cách tính toán độ tin cậy của mỗi cửa sổ con trong khung hình.")
    st.markdown(
        "5. **Theo Dõi Đối Tượng:** Quá trình lặp lại cho đến khi kết thúc video.")

    st.markdown("### 3. Ví dụ minh họa")

    # st.video('servicess/Instance_Search/output_tracked_video.avi')
    @st.cache_data
    def download_video(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            st.error("Không thể tải video.")

    # URL của video
    video_url = "https://github.com/lbnm203/OpenCV-App/blob/master/servicess/Instance_Search/output_tracked_video.avi"
    video_data = download_video(video_url)

    if video_data:
        st.video(video_data)
    # video_url = "https://github.com/lbnm203/OpenCV-App/blob/master/servicess/Instance_Search/output_tracked_video.avi"
    # # Tải video và lưu tạm vào bộ nhớ
    # response = requests.get(video_url)
    # if response.status_code == 200:
    #     with open("output_tracked_video.avi", "wb") as f:
    #         f.write(response.content)
    #     # Sử dụng file đã tải về để hiển thị trong Streamlit
    #     with open("output_tracked_video.avi", "rb") as video_file:
    #         video_bytes = video_file.read()
    #         st.video(video_bytes)

    st.markdown(
    "### 4. Các thách thức trong Object Tracking với Boosting Tracker")

    data_challenges = {
        "Thách thức": [
            "Background clutters",
            "Illumination variations",
            "Occlusion",
            "Fast motion"
        ],
        "Tình huống lỗi": [
            "Nền có nhiều đối tượng tương tự gây nhầm lẫn, hoặc vật thể hòa trộn vào nền.",
            "Khi điều kiện ánh sáng thay đổi đột ngột (sáng/tối) hoặc màu sắc của đối tượng thay đổi.",
            "Khi vật thể bị che khuất hoàn toàn hoặc một phần bởi các đối tượng khác trong khung hình.",
            "Khi vật thể di chuyển nhanh, dẫn đến motion blur hoặc đối tượng di chuyển ra ngoài khung hình."
        ],
        "Khả năng xử lý của Boosting Tracker": [
            "Boosting Tracker có thể nhầm lẫn giữa đối tượng và nền nếu bộ phân loại không đủ mạnh hoặc không được cập nhật đủ tốt.",
            "Boosting Tracker gặp khó khăn khi thay đổi độ sáng nhanh chóng vì bộ phân loại có thể không thích ứng đủ nhanh với điều kiện mới.",
            "Boosting Tracker dễ bị mất dấu vật thể khi bị che khuất, vì bộ phân loại không có khả năng \"ghi nhớ\" đặc tính của đối tượng.",
            "Boosting Tracker gặp khó khăn với chuyển động nhanh, đặc biệt là khi không đủ thời gian để cập nhật hoặc theo dõi trong khung tiếp theo."
        ]
    }

    challenges_df = pd.DataFrame(data_challenges)
    st.table(challenges_df)
