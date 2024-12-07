import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Model yükleme
model_path = "/Users/sidar/Downloads/best.pt"  # YOLOv8 ağırlık dosyanızın yolu
model = YOLO(model_path)

# Arayüz Başlığı
st.set_page_config(layout="wide")
st.title("YOLOv8 ile Nesne Tespiti")
st.subheader("Konu: Görüntüleri işleyerek yeni görüntü üzerinden tahmin", divider="green")

# Sol panel: Eşik değerleri, tahmin sonuçları ve performans görseli
with st.sidebar:
    st.header("Ayarlar")
    confidence_threshold = st.slider("Confidence Threshold:", 0, 100, 50, step=1) / 100
    overlap_threshold = st.slider("Overlap Threshold:", 0, 100, 50, step=1) / 100

    st.header("Tahmin Edilen Nesneler")

    # Performans görseli alt kısımda
    st.header("Performans")
    with st.expander("Tıklayın: Model Performansı Görseli"):
        # Model performans görselini yükle ve göster
        model_image_path = "/Users/sidar/Downloads/results.png"  # Model görsel yolu
        if os.path.exists(model_image_path):
            st.image(model_image_path, caption="Model Performansı")
        else:
            st.warning("Performans görseli bulunamadı. Lütfen doğru bir yol belirtin.")

# Görüntü yükleme
uploaded_file = st.file_uploader("Bir görüntü seçin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Yüklenen görüntüyü işleme ve RGB formatına dönüştürme
    image = Image.open(uploaded_file).convert("RGB")  # Görüntüyü RGB formatına dönüştür

    # YOLO tahmini
    results = model.predict(source=np.array(image), conf=confidence_threshold, iou=overlap_threshold)
    detections = results[0].boxes

    # Tahmin edilen nesneleri çizim için hazırlık
    output_image = Image.fromarray(np.array(image))
    draw = ImageDraw.Draw(output_image)

    # Yazı tipi ayarları
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Sisteminizdeki yazı tipini seçin
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 20)
    else:
        font = ImageFont.load_default()

    predictions = []
    for box in detections:
        cls = int(box.cls[0])
        confidence = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Sınıf ve güven skorunu ekle
        predictions.append({
            "class": model.names[cls],
            "confidence": f"{confidence:.2f}",
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

        # Görüntü üzerine kutu çiz
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Metni kutunun üst kısmına veya içerisine yaz
        text = f"{model.names[cls]} ({confidence:.2f})"

        # Text bbox ile metin kutusunu al
        text_bbox = draw.textbbox((x1, y1), text, font=font)  # Metin kutusunun boyutlarını al
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x, text_y = x1, y1 - text_height - 5

        # Eğer metin üstte taşarsa bounding box içine yerleştir
        if text_y < 0:
            text_y = y1 + 5

        # Metin arka planı
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill="red"
        )
        # Metin yaz
        draw.text((text_x, text_y), text, fill="white", font=font)

    # Görüntüyü orta boyutta göstermek için boyutlandırma
    desired_width = 640  # Sabit genişlik
    aspect_ratio = output_image.height / output_image.width
    resized_image = output_image.resize((desired_width, int(desired_width * aspect_ratio)), Image.LANCZOS)

    # Sağda görüntüyü göster
    st.image(resized_image, caption="Tahmin Sonucu", use_column_width=False)

    # Solda tahmin edilen nesneleri göster
    with st.sidebar:
        for prediction in predictions:
            st.write(f"- **{prediction['class']}** (Güven: {prediction['confidence']})")
else:
    st.info("Lütfen bir görüntü yükleyin.")

# Yapanların bilgileri
st.markdown("---")
st.markdown("### Yapanlar")
st.markdown("""
**Sidar Deniz Topaloğlu**  
[LinkedIn](https://www.linkedin.com/in/sidar-deniz-topaloğlu/) 

**Melike Zeynep Işıktaş**  
[LinkedIn](https://www.linkedin.com/in/melike-zeynep-işıktaş-182a5b285/)   
""")
