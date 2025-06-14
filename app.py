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
import streamlit.components.v1 as components
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def download_model_if_needed():
    model_path = "Xception_banh_model.keras"
    if not os.path.exists(model_path):
        file_id = "1SW5eLCyuDK4n1hOTCedBbKrtAjpEuNaC"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

# Tải model nếu cần
download_model_if_needed()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Bán kính Trái Đất (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # kết quả trả về tính bằng km

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

    pdf.ln(50)

    pdf.set_font("DejaVu", 'B', 12)
    pdf.cell(0, 10, "Biểu đồ xác suất", ln=1)

    current_y = pdf.get_y()
    pdf.image(bar_path, x=30, y=current_y, w=140) 

    pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(pdf_output.name)
    pdf_output.seek(0)
    return pdf_output

# Giao diện Streamlit
st.set_page_config(page_title="Phân loại bánh", layout="wide")

st.markdown("""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
html, body, [class*="css"] {
    font-family: 'Quicksand', sans-serif;
}
img {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    max-width: 100%;
    height: auto;
}
h1 {
    color: #e65100;
    font-size: 2.2rem;
    font-weight: 700;
}
h2, h3 {
    color: #3F51B5;
    margin-top: 0.5rem;
}
.stAlert {
    border-radius: 12px;
    border-left: 6px solid #FF5722;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    font-size: 1rem;
}
section[data-testid="stSidebar"] {
    border-radius: 0 12px 12px 0;
    background-color: #ffffff;
    box-shadow: 2px 0 8px rgba(0,0,0,0.05);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
/* Giao diện gọn gàng hơn trên điện thoại */
    @media screen and (max-width: 600px) {
        .stAlert {
            font-size: 0.95rem;
        }
        h1 {
            font-size: 1.5rem;
        }
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
recipe_assets = {
    "Cheesecake": {
        "pdf": "https://hongminh1602.github.io/cake-xception-app/recipes/chesecake_recipe.pdf",
        "video": "https://www.youtube.com/watch?v=aMBecr0SJ8I&pp=ygUkaMaw4bubbmcgZOG6q24gbMOgbSBiw6FuaCBjaGVlc2VjYWtl"
    },
    "Donut": {
        "pdf": "https://hongminh1602.github.io/cake-xception-app/recipes/donut_recipe.pdf",
        "video": "https://www.youtube.com/watch?v=zMkLRWjahOk&pp=ygUfaMaw4bubbmcgZOG6q24gbMOgbSBiw6FuaCBkb251dNIHCQneCQGHKiGM7w%3D%3D"
    },
    "Macaron": {
        "pdf": "https://hongminh1602.github.io/cake-xception-app/recipes/macaron_recipe.pdf",
        "video": "https://www.youtube.com/watch?v=MFyc72Bfqbs&pp=ygUhaMaw4bubbmcgZOG6q24gbMOgbSBiw6FuaCBtYWNhcm9u"
    },
    "Tiramisu": {
        "pdf": "https://hongminh1602.github.io/cake-xception-app/recipes/tiramisu_recipe.pdf",
        "video": "https://www.youtube.com/watch?v=vF54bj3V5Es"
    }
}
# Gợi ý tiệm bánh theo từng loại (dùng dữ liệu giả định)
locations = {
    "Cheesecake": [
    {
        "name": "Reverie Dessert ( bán online ) ",
        "lat": 0,  
        "lon": 0,
    }
    ],
    "Donut": [
    {
        "name": "Giản Donuts",
        "lat": 21.018940671560365,  
        "lon": 105.84002679999999,
        "map_url": "https://maps.app.goo.gl/S2khYBf9hSV4zCQv9?g_st=iz"
    },
    {
        "name": "Lịm Donuts Hanoi",
        "lat": 221.03075186295445, 
        "lon": 105.84661844047851,
        "map_url": "https://maps.app.goo.gl/oNHR1FLvwJ83FJST8?g_st=iz"
    }
    ],
    "Macaron": [
    {
        "name": "La Rosette Macaron",
        "lat": 20.995968424740052, 
        "lon": 105.82483821986816,
        "map_url": "https://maps.app.goo.gl/Y6bQkQJZ2jfbtXN66"
    },
    {
        "name": "Ryan's Patisserie",
        "lat": 21.011402225563693,
        "lon": 105.81893313068578,
        "map_url": "https://maps.app.goo.gl/utpgS6Hm6xM8UKrJ8?g_st=iz"
    }
    ],
    "Tiramisu": [
    {
        "name": "Kat's Bakery",
        "lat": 21.0183411,
        "lon": 105.828166,
        "map_url": "https://www.google.com/maps/dir/20.925027,105.8642597/21.0183411,105.828166/"
    },
    {
        "name": "Cake by Xuf",
        "lat": 21.010030892626265,
        "lon": 105.83136459052602,
        "map_url": "https://maps.app.goo.gl/oxBQ2XRJxgx7mMJD7?g_st=iz"
    }
    ]
}
def predict(img):
    img = img.resize((299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return preds, pred_class, confidence

# Sidebar
with st.sidebar.expander("📘**Thông tin nhóm**"):
    st.markdown("👥 **Nhóm:** 14")
    st.markdown("👨‍🏫 **GVHD:** Thầy Vũ Trọng Sinh")
    st.markdown("🏫 **Lớp:** 242IS54A01")
    st.markdown("📚 **Môn:** Trí tuệ nhân tạo")
    st.sidebar.markdown("---")

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

if uploaded_file:
    img = Image.open(uploaded_file)

    img_col, result_col = st.columns([1, 1])
    with img_col:
        st.image(img, caption="📸 Ảnh bạn vừa tải", use_container_width=True)

    preds, pred_class, confidence = predict(img)

    with result_col:
        st.markdown("### 🔍 Kết quả dự đoán:")
        st.markdown(f"👉 **{pred_class}** với độ tin cậy **{confidence*100:.2f}%**")
        st.info(descriptions[pred_class])

    # ✅ VẼ BIỂU ĐỒ CHỈ NẾU ĐÃ TẢI ẢNH
    st.markdown("### 📊 Biểu đồ xác suất")

    col_left, col_chart, col_right = st.columns([0.3, 6, 0.3])
    with col_chart:
        fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
        y_pos = np.arange(len(class_names))
        ax1.barh(y_pos, preds, align='center', color=["#FFC107", "#FF5722", "#9C27B0", "#3F51B5"])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(class_names, fontsize=11)
        ax1.invert_yaxis()
        ax1.set_xlabel('Xác suất', fontsize=11)
        ax1.set_xlim(0, 1.0)
        ax1.set_title('Phân bố xác suất các loại bánh', fontsize=13)

        for i, v in enumerate(preds):
            ax1.text(v + 0.01, i, f"{v*100:.2f}%", va='center', fontsize=10)

        st.pyplot(fig1)

    # ✅ TẠO FILE PDF CHỈ NẾU ĐÃ CÓ KẾT QUẢ
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

    # ✅ Địa chỉ mua 
    if pred_class in locations:
        st.markdown("### 📍 Gợi ý địa điểm mua bánh")
    
        df_map = pd.DataFrame([{
            "latitude": loc["lat"],
            "longitude": loc["lon"]
        } for loc in locations[pred_class]])
        st.map(df_map)
    
        # Vị trí người dùng (giả định)
        user_lat, user_lon = 21.00892346213516, 105.82871755204096

        for item in locations[pred_class]:
            distance_km = haversine(user_lat, user_lon, item['lat'], item['lon'])
            st.markdown(f"**🍰 {item['name']}** – 📍 Cách bạn khoảng **{distance_km:.2f} km**")
            
            if 'map_url' in item:
                st.markdown(f"[🗺️ Xem đường đi trên Google Maps]({item['map_url']})", unsafe_allow_html=True)
    
    # ✅ Xem công thức và video hướng dẫn 
    with st.expander("📖 Xem công thức và hướng dẫn chi tiết"):
        st.markdown("#### 🎥 Video hướng dẫn:")
        st.markdown(
            f'<a href="{recipe_assets[pred_class]["video"]}" target="_blank">👉 Xem video hướng dẫn </a>',
            unsafe_allow_html=True
        )
        
        st.markdown("#### 📄 Công thức chi tiết (PDF):")
        st.markdown(
            f'<a href="{recipe_assets[pred_class]["pdf"]}" target="_blank">👉 Mở công thức dạng PDF</a>',
            unsafe_allow_html=True
        )
