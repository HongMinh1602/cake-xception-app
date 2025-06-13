import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from fpdf import FPDF
import tempfile
import io
import os
import gdown

def download_model_if_needed():
    model_path = "Xception_banh_model.keras"
    if not os.path.exists(model_path):
        file_id = "1SW5eLCyuDK4n1hOTCedBbKrtAjpEuNaC"  # Thay ID bằng ID thật của bạn
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# Tải model nếu chưa có
download_model_if_needed()

def create_pdf(image_path, pred_class, confidence, preds, class_names, bar_fig):
    pdf = FPDF()
    pdf.set_left_margin(15)
    pdf.set_right_margin(20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    font_folder = os.getcwd()
    pdf.add_font("DejaVu", "", os.path.join(font_folder, "DejaVuSans.ttf"), uni=True)
    pdf.add_font("DejaVu", "B", os.path.join(font_folder, "DejaVuSans-Bold.ttf"), uni=True)

    pdf.set_font("DejaVu", 'B', 16)
    pdf.cell(0, 10, txt="BÁO CÁO PHÂN LOẠI BÁNH", ln=True, align="C")
    pdf.ln(10)

    tmp_dir = tempfile.mkdtemp()
    input_img_path = os.path.join(tmp_dir, "input.jpg")
    bar_path = os.path.join(tmp_dir, "bar.png")

    image_path.save(input_img_path)
    bar_fig.savefig(bar_path, bbox_inches='tight')

    pdf.set_font("DejaVu", 'B', 12)
    pdf.cell(90, 10, "Ảnh đầu vào", ln=0)
    pdf.cell(0, 10, "Kết quả dự đoán", ln=1)
    pdf.image(input_img_path, x=15, y=pdf.get_y(), w=60)

    y_start = pdf.get_y()
    pdf.set_xy(80, y_start)
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8, f"Dự đoán: {pred_class} ({confidence*100:.2f}%)\n\n" +
                     "\n".join([f"- {cls}: {prob*100:.2f}%" for cls, prob in zip(class_names, preds)]))

    pdf.ln(45)

    pdf.set_font("DejaVu", 'B', 12)
    pdf.cell(0, 10, "Biểu đồ phân bố xác suất", ln=1)
    current_y = pdf.get_y()
    pdf.image(bar_path, x=15, y=current_y, w=160)

    pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(pdf_output.name)
    pdf_output.seek(0)
    return pdf_output

st.set_page_config(page_title="Phân loại bánh", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Quicksand', sans-serif; }
    .appview-container { background-color: #f5f7fa; padding: 1rem; }
    img { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 100%; height: auto; }
    .stAlert { border-radius: 12px; border-left: 6px solid #FF5722; box-shadow: 0 3px 10px rgba(0,0,0,0.08); font-size: 1rem; }
    h1 { color: #e65100; font-size: 2.2rem; font-weight: 700; }
    h3, h2 { color: #3F51B5; margin-top: 0.5rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; }
    section[data-testid="stSidebar"] { border-radius: 0 12px 12px 0; background-color: #ffffff; box-shadow: 2px 0 8px rgba(0,0,0,0.05); }
    figcaption { text-align: center; color: #666; font-size: 0.9rem; }
    @media screen and (max-width: 600px) {
        .stAlert { font-size: 0.95rem; }
        h1 { font-size: 1.5rem; }
    }
    </style>
""", unsafe_allow_html=True)

model = load_model("Xception_banh_model.keras")
class_names = ['Cheesecake', 'Donut', 'Macaron', 'Tiramisu']

descriptions = {
    "Cheesecake": "🧀 Cheesecake là loại bánh ngọt mềm mịn làm từ kem phô mai, thường có đế là bánh quy nghiền.",
    "Donut": "🍩 Donut là bánh vòng chiên, thường được phủ socola, đường hoặc topping trang trí nhiều màu.",
    "Macaron": "🌈 Macaron là bánh hạnh nhân Pháp, vỏ giòn tan, bên trong mềm mịn, nhiều màu sắc đẹp mắt.",
    "Tiramisu": "☕ Tiramisu là bánh Ý đặc trưng với vị cà phê, kem mascarpone và lớp cacao phủ bên trên."
}

def predict(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return preds, pred_class, confidence

with st.sidebar.expander("📘**Thông tin nhóm**"):
    st.markdown("👥 **Nhóm:** 14")
    st.markdown("👨‍🏫 **GVHD:** Thầy Vũ Trọng Sinh")
    st.markdown("🏫 **Lớp:** 242IS54A01")
    st.markdown("📚 **Môn:** Trí tuệ nhân tạo")

with st.sidebar.expander("🧠 Giới thiệu model Xception"):
    st.markdown("""
    1. **Xception** là viết tắt của *Extreme Inception* – một mô hình mạng nơ-ron tích chập (CNN) nâng cấp từ Inception.
    2. Thay vì dùng các khối tích chập tiêu chuẩn, Xception sử dụng **Depthwise Separable Convolution** để tăng hiệu quả tính toán.
    3. Mô hình này có **hiệu suất cao** trong phân loại ảnh, đặc biệt tốt khi áp dụng cho các tập dữ liệu hình ảnh có chi tiết đặc trưng như bánh ngọt.
    4. Nhóm đã fine-tune Xception để phân biệt giữa 4 loại bánh: *Cheesecake, Donut, Macaron, Tiramisu*.
    """)

with st.sidebar.expander("📊 Model hoạt động như thế nào"):
    st.markdown("""
    ### 🔍 Cách hoạt động của mô hình:
    1. 🖼 Tải ảnh bánh lên ứng dụng.
    2. 📏 Ảnh được resize về **299x299** pixel và chuẩn hóa dữ liệu.
    3. 🤖 Mô hình Xception (Deep Learning) xử lý ảnh để trích xuất đặc trưng.
    4. 📈 Mô hình tính toán xác suất thuộc về từng loại bánh.
    5. ✅ Kết quả cuối cùng là loại bánh có xác suất cao nhất.
    """)

st.title("🎂 Phân loại bánh với mô hình Xception")

uploaded_file = st.file_uploader("📷 Tải ảnh bánh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_col, result_col = st.columns([1, 1])

    with img_col:
        st.image(img, caption="📸 Ảnh bạn vừa tải", use_container_width=True)

    preds, pred_class, confidence = predict(img)

    with result_col:
        st.markdown('<div style="padding-top: 150px;">', unsafe_allow_html=True)
        st.markdown("### 🔍 Kết quả dự đoán:")
        st.markdown(f"👉 **{pred_class}** với độ tin cậy **{confidence*100:.2f}%**")
        st.info(descriptions[pred_class])

    st.markdown("### 📊 Biểu đồ xác suất")
    fig1, ax1 = plt.subplots()
    y_pos = np.arange(len(class_names))
    colors = {"Cheesecake": "#FFC107", "Donut": "#FF5722", "Macaron": "#9C27B0", "Tiramisu": "#3F51B5"}
    bar_colors = [colors[name] for name in class_names]
    bars = ax1.barh(y_pos, preds, align='center', color=bar_colors)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(class_names)
    ax1.invert_yaxis()
    ax1.set_xlabel('Xác suất')
    ax1.set_xlim(0, 1.0)
    ax1.set_title('Phân bố xác suất các loại bánh')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{preds[i]*100:.2f}%", va='center')

    st.pyplot(fig1)

    pdf_filename = st.text_input("📄 Đặt tên file PDF (không cần .pdf)", value="bao_cao_du_doan_banh")
    if st.button("📄 Lưu kết quả dạng PDF"):
        pdf_file = create_pdf(img, pred_class, confidence, preds, class_names, fig1)
        with open(pdf_file.name, "rb") as f:
            st.download_button(
                label="📥 Tải file PDF",
                data=f,
                file_name=f"{pdf_filename}.pdf",
                mime="application/pdf"
            )
