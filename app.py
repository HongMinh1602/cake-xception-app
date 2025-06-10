import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load mô hình
model = tf.keras.models.load_model('model/xception_banh_model.keras')

# Danh sách nhãn (tùy mô hình bạn huấn luyện)
class_names = ['Bánh bông lan', 'Bánh su kem', 'Bánh tart', 'Bánh mì', 'Bánh cupcake']

# Hàm xử lý ảnh
def preprocess_image(img):
    img = img.resize((299, 299))  # Kích thước của Xception
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Giao diện
st.title("🍰 Ứng dụng nhận diện bánh ngọt")
st.write("Tải ảnh lên và xem dự đoán của mô hình")

uploaded_file = st.file_uploader("Chọn ảnh bánh", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã tải", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✨ Mô hình dự đoán: **{predicted_class}**")
