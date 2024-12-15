import streamlit as st
import cv2 as cv
from PIL import Image, ImageOps
from tensorflow.keras import layers, models
import pickle
import numpy as np
from streamlit_drawable_canvas import st_canvas
# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F

# st.set_page_config(
#     page_title="🎈Hand Writing Regconition Applications",
#     # page_icon=Image.open("./images/Logo/logo_welcome.png"),
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


def predict_with_image(image):
    model = models.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2)))

    # Flattening Layer
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(100))
    model.add(layers.ReLU())

    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))

    loaded_weights = []
    with open("./servicess/Hand_Writte_Reg/cnn_mnist_model_v2.pth", "rb") as file:
        loaded_weights = pickle.load(file)

    model.set_weights(loaded_weights)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Process image
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_cpy = image.copy()
    image[image_cpy != 0] = 255
    # st.image(image)
    image = cv.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    # Reshape ((batch_size, height, width, channels))
    image = image.reshape((1, 28, 28, 1))
    predictions = model.predict(image)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels


# @st.cache_resource
# def load_model(model_path, num_classes=10):
#     cnn_model = predict_with_image(num_classes)
#     cnn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
#     cnn_model.eval()
#     return cnn_model

# model = load_model('servicess/Hand_Writte_Reg/cnn_mnist_model_v2.pt')


def crop_and_center_mnist(img):
    img_cpy = img.copy()
    id_white = np.where(img_cpy > 0)

    x_min = np.min(id_white[1])
    x_max = np.max(id_white[1])
    y_min = np.min(id_white[0])
    y_max = np.max(id_white[0])

    height = y_max - y_min
    width = x_max - x_min

    # Bounding box
    x, y, w, h = (x_min, y_min, width, height)
    cropped_img = img[y: y + h, x: x+w]

    padding = int(h * 0.25)

    # 5. Thêm padding
    padded_img = cv.copyMakeBorder(
        cropped_img,
        padding,
        padding,
        padding,
        padding,
        cv.BORDER_CONSTANT,
        value=0,
    )

    # 6. Resize về 28x28
    if padded_img.shape[0] <= 0 or padded_img.shape[1] <= 0:
        st.warning("Không tìm thấy hình vẽ!")
        return np.zeros((28, 28), dtype=np.uint8)

    final_img = cv.resize(padded_img, (28, 28), interpolation=cv.INTER_AREA)
    return final_img


def Applications():
    undo_symbol = "↩️"
    trash_symbol = "🗑️"
    st.markdown('<span style = "color:blue; font-size:24px;">Cách sử dụng</span>',
                unsafe_allow_html=True)
    st.write(
        "  - Vẽ chữ số cần dự đoán **(0, 1, ...9)** lên hình chữ nhật màu đen ở dưới")
    st.write(
        f"  - Khi cần hoàn tác thao tác vừa thực hiện, **Click** chuột vào {undo_symbol} ở dưới ảnh")
    st.write(
        f"  - Khi cần Reset lại từ đầu các thao tác, **Click** chuột vào {trash_symbol} ở dưới ảnh")
    st.write("  - Sau đó nhấn nút **Submit** ở bên dưới để nhận kết quả dự đoán")
    st.write("**Lưu ý:** Chỉ vẽ một chữ số **duy nhất**")
    stroke_width = 3
    stroke_color = "red"
    drawing_mode = "freedraw"
    image = Image.new("RGB", (280, 280), "black")
    c = st.columns([3, 7])
    with c[0]:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#000000",
            background_image=image,
            # update_streamlit=realtime_update,
            width=image.width,
            height=image.height,
            drawing_mode=drawing_mode,
            key="Handwriting Letter Recognize",
        )
        c[1].markdown("**Kết quả dự đoán**")
        if st.button("Submit"):
            if canvas_result.image_data is not None:
                image_canvas = canvas_result.image_data
                if image_canvas.dtype != np.uint8:
                    image_canvas = cv.normalize(
                        image_canvas, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                image_canvas = cv.cvtColor(image_canvas, cv.COLOR_BGR2GRAY)
                image_cpy = image_canvas.copy()
                image_canvas[image_cpy != 0] = 255
                # st.image(image_canvas)
                image_crop = crop_and_center_mnist(image_canvas)
                # st.image(image_crop)
                results = predict_with_image(image_crop)
                c[1].markdown(
                    f"<p style='font-size: 30px;'>{results[0]}</p>", unsafe_allow_html=True)
            else:
                st.warning(
                    "Vui lòng vẽ kí tự cần dự đoán trước khi **Submit!**")


def Results():
    c = st.columns(2)
    with c[0]:
        image_accuracy = cv.imread("./servicess/Hand_Writte_Reg/acc_cnn.png")
        st.image(image_accuracy,
                 caption="Training and Validation Acccuracy", channels="BGR")
    with c[1]:
        image_loss = cv.imread("./servicess/Hand_Writte_Reg/loss_cnn.png")
        st.image(image_loss, caption="Training and Validation Loss", channels="BGR")

    accuracy_test = 0.91
    st.markdown(f"- Độ chính xác cho ra trên tập test là: **{accuracy_test}%**")


def Training():
    st.header("3. Thiết lập quá trình huấn luyện")

    st.markdown(
        """
            - Ta sẽ dùng
                - 50000 ảnh cho tập huấn luyện (train)
                - 10000 ảnh cho tập kiểm thử (valid)
                - 10000 ảnh cho tập test.
                - Hàm tối ưu: Adam
                - Hàm mất mát: CrossEntropyLoss
                - Độ đo: Accuracy
                - Learning_rate: 0.001
                - Số lượng Epoch: 10
                """
    )


def Text():
    st.title(" ✨ Handwriting Letter Recognize Application")
    st.header("1. Tập dữ liệu MNIST")
    st.markdown(
        """
        - Bộ dữ liệu MNIST là một tập hợp các hình ảnh thang độ xám của các chữ số viết tay (0-9). 
        Mỗi hình ảnh có kích thước 28x28 pixel và bộ dữ liệu chứa 70.000 mẫu trong đó có 60.000 mẫu training và 10.000 mẫu testing.
        - Bộ dữ liệu MNIST gồm có 10 nhãn (label) là chứ số từ 0 -> 9
                """)

    st.markdown("Dưới đây là một số ảnh minh họa dataset **MNIST**")
    c = st.columns([2, 6, 2])
    image = cv.imread("./servicess/Hand_Writte_Reg/dataset_visualize.png")
    c[1].image(image, channels="BGR", caption="Minh họa dataset MNIST")
    st.header("2. Kiến trúc mô hình")

    image_achitecture = cv.imread("./servicess/Hand_Writte_Reg/cnn_architecture.png")
    cc = st.columns([3, 5, 2])
    cc[1].image(image_achitecture, caption="Kiến trúc của mô hình", channels="BGR")
    st.markdown("#### 2.1 Giải thích kiến trúc trên")
    st.markdown("""
        - Convolutional Layer 1:
            - Chi tiết: 32 filters, kernel 3x3, padding = 1, tiếp theo là ReLU Activation.
            - Mục đích: Lớp này trích xuất các đặc trưng cấp thấp như cạnh và góc từ hình ảnh đầu vào. Kích hoạt ReLU đưa vào tính phi tuyến tính, cho phép mô hình học các mẫu phức tạp.

        - Max Pooling 1:
            - Chi tiết: kernel 2x2, stride = 2.
            - Mục đích: Giảm kích thước không gian của bản đồ đặc trưng, giảm chi phí tính toán và kiểm soát tình trạng quá khớp. Nó giữ lại các đặc trưng quan trọng trong khi loại bỏ các chi tiết không liên quan.

        - Convolutional Layer 2:
            - Chi tiết: 64 filters, kernel 3x3, padding = 1, tiếp theo là ReLU Activation.
            - Mục đích: Trích xuất các đặc trưng cấp cao hơn, chẳng hạn như kết cấu hoặc các phần của hình dạng chữ số, dựa trên các đặc trưng đã học được trong lớp tích chập đầu tiên.

        - Max Pooling 2:
        Chi tiết: kernel 2x2, stride = 2.
        Mục đích: Giảm thêm kích thước không gian của bản đồ đặc trưng, đảm bảo biểu diễn nhỏ gọn.

        - Flatten Layer::
            - Chi tiết: Chuyển đổi bản đồ đặc trưng 2D thành vectơ đặc trưng 1D.
            - Mục đích: Chuẩn bị dữ liệu cho các lớp được kết nối đầy đủ, cho phép mô hình thực hiện phân loại.

        - Fully Connected Layer 1:
            - Chi tiết: 7x7x64 (3136) đặc trưng đầu vào được kết nối với 100 nơ-ron, sau đó ReLU Activation.
            - Mục đích: Kết hợp các đặc trưng đã trích xuất và học các kết hợp phức tạp để phân loại. Kích hoạt ReLU đảm bảo tính phi tuyến tính.

        - Fully Connected Layer 2 (Output Layer):
            - Chi tiết: 100 nơ-ron được kết nối với 10 nơ-ron đầu ra (một cho mỗi lớp chữ số).
            - Mục đích: Tạo điểm phân loại cuối cùng cho mỗi lớp chữ

    """)

    st.markdown("#### 2.2 Lý do chọn kiến trúc trên")
    st.markdown(
        """
                Kiến trúc này được chọn vì sự cân bằng giữa tính đơn giản và hiệu quả trong việc giải quyết các nhiệm vụ nhận dạng chữ số viết tay. 
                Kiến trúc này sử dụng hai lớp tích chập với kích hoạt ReLU để trích xuất các đặc trưng cấp, sau đó là max pooling để giảm chiều 
                không gian và độ phức tạp của tính toán. Các lớp được kết nối đầy đủ kết hợp các đặc trưng này để các dự đoán mạnh mẽ, với dropout 
                thêm chính quy hóa để giảm thiểu tình trạng quá khớp. Thiết kế này rất phù hợp với tập dữ liệu MNIST, tận dụng độ phân giải thấp của nó 
                (hình ảnh thang độ xám 28x28) để đạt được độ chính xác cao trong khi vẫn duy trì hiệu quả tính toán.
                """
    )
    Training()
    st.header(" 4. Kết quả thực nghiệm")
    Results()
    # st.markdown("#### 4. Ứng dụng")
    # Applications()


# def App():
#     Text()


# App()
