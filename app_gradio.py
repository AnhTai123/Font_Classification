#!/usr/bin/env python3
"""
Gradio app: Font classification + EasyOCR.
- Upload ảnh → EasyOCR nhận dạng chữ trong ảnh.
- Dự đoán Top 3 font, ảnh kết quả (ảnh gốc + 3 bảng chữ cái), tải ZIP 3 font.

Chạy:
  python app_gradio.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import torch
import gradio as gr

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Import logic từ api_inference (model, transform, inference, HDF5)
from api_inference import (
    MODEL_PATH,
    FONT_DIR,
    HDF5_PATH,
    TOP_K,
    device,
    build_class_mapping,
    load_font_mapping,
    load_trained_model,
    get_font_file_path,
    create_result_image,
    create_zip_in_memory,
    predict_tensor,
    run_ocr_on_image_with_boxes,
    segment_then_classify_image,
    val_transform,
    Prediction,
)

# Global: model, mapping, EasyOCR reader
model = None
index_to_label = None
font_names_hdf5 = None
font_files_hdf5 = None
ocr_reader = None


def load_resources():
    """Tải model và mapping (chạy 1 lần khi launch app)."""
    global model, index_to_label, font_names_hdf5, font_files_hdf5
    if model is not None:
        return
    print("Đang tải model và tài nguyên...")
    # 1. Mapping từ checkpoint hoặc FONT_DIR
    try:
        checkpoint_data = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint_data, dict) and "index_to_label" in checkpoint_data:
            index_to_label = checkpoint_data["index_to_label"]
            num_classes = checkpoint_data.get("num_classes")
            print(f"OK Mapping từ checkpoint: {num_classes} classes")
        else:
            index_to_label, num_classes = build_class_mapping(FONT_DIR)
    except Exception as e:
        print(f"[!] Load mapping: {e}")
        index_to_label, num_classes = build_class_mapping(FONT_DIR)
    # 2. HDF5 (names, files)
    if os.path.exists(HDF5_PATH):
        font_names_hdf5, font_files_hdf5 = load_font_mapping(HDF5_PATH)
        if font_names_hdf5:
            print(f"OK HDF5: {len(font_names_hdf5)} fonts")
    else:
        font_names_hdf5, font_files_hdf5 = None, None
    # 3. Model
    model_loaded, ckpt_idx_to_label, _ = load_trained_model(
        MODEL_PATH, len(index_to_label), device
    )
    if ckpt_idx_to_label is not None:
        index_to_label = ckpt_idx_to_label
    model = model_loaded.to(device).eval()
    print("OK Model sẵn sàng.")


def load_ocr_reader():
    """Khởi tạo EasyOCR reader (lazy, 1 lần). Hỗ trợ tiếng Anh + tiếng Việt."""
    global ocr_reader
    if ocr_reader is not None:
        return ocr_reader
    if not EASYOCR_AVAILABLE:
        return None
    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(["en", "vi"], gpu=use_gpu, verbose=False)
    return ocr_reader


def run_ocr(image):
    """Chạy EasyOCR trên ảnh (numpy RGB). Trả về chuỗi mô tả text đã nhận dạng."""
    if not EASYOCR_AVAILABLE:
        return "Cần cài EasyOCR: pip install easyocr"
    if image is None or image.size == 0:
        return ""
    try:
        reader = load_ocr_reader()
        # EasyOCR đọc ảnh BGR hoặc RGB
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
        results = reader.readtext(img_bgr)
        if not results:
            return "Không nhận dạng được chữ trong ảnh."
        lines = []
        for i, (bbox, text, conf) in enumerate(results, 1):
            lines.append(f"{i}. {text.strip()} ({conf:.2f})")
        return "\n".join(lines)
    except Exception as e:
        return f"Lỗi OCR: {str(e)}"


def _crop_bbox(rgb, bbox, padding=12):
    """Crop ảnh theo bbox (x_min, y_min, x_max, y_max), thêm padding. Trả về numpy RGB."""
    h, w = rgb.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    crop = rgb[y_min:y_max, x_min:x_max].copy()
    if crop.size == 0:
        return None
    return crop


def predict(image):
    """
    OCR trước → với mỗi vùng chữ OCR thấy được: crop vùng đó, predict font, tạo ảnh kết quả (crop + Top 3 bảng chữ cái).
    Đầu ra: N ảnh (N = số vùng chữ), mỗi ảnh = vùng chữ đã crop + 3 font.
    """
    if image is None:
        return None, "Vui lòng tải ảnh lên.", None, ""
    load_resources()
    global model, index_to_label, font_names_hdf5, font_files_hdf5
    try:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        original_rgb = image.copy()
        # 1) OCR để lấy các vùng chữ (bbox + text)
        ocr_results = run_ocr_on_image_with_boxes(original_rgb)
        ocr_text = "\n".join(
            f"{i+1}. {r['text']} ({r['confidence']:.2f})"
            for i, r in enumerate(ocr_results)
        ) if ocr_results else "Không nhận dạng được chữ trong ảnh."
        # 2) Nếu không có vùng nào: dùng cả ảnh làm 1 vùng (fallback)
        if not ocr_results:
            ocr_results = [{"bbox": (0, 0, original_rgb.shape[1], original_rgb.shape[0]), "text": "(cả ảnh)", "confidence": 0.0}]
        gallery_list = []
        first_results = None
        for idx, ocr_item in enumerate(ocr_results):
            crop_rgb = _crop_bbox(original_rgb, ocr_item["bbox"])
            if crop_rgb is None or crop_rgb.size == 0:
                continue
            # Pad 224x224 → segment → mask * ảnh → kéo sáng → dùng ảnh đó để classification
            image_for_clf = segment_then_classify_image(crop_rgb)  # (224,224,3) uint8
            augmented = val_transform(image=image_for_clf)
            tensor = augmented["image"].unsqueeze(0)
            results = predict_tensor(model, tensor, device, index_to_label, TOP_K)
            if not results:
                continue
            results = results[:TOP_K]
            if first_results is None:
                first_results = results
            # Hiển thị ảnh đã segment + brighten (ảnh đưa vào classification)
            result_image_bytes = create_result_image(
                image_for_clf, results, font_names_hdf5, font_files_hdf5
            )
            if result_image_bytes:
                result_img = np.frombuffer(result_image_bytes, np.uint8)
                result_img = cv2.imdecode(result_img, cv2.IMREAD_COLOR)
                if result_img is not None:
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    caption = f'"{ocr_item["text"]}" — Top 3 font (bấm vào ảnh để xem phóng to)'
                    gallery_list.append((result_img, caption))
        if not gallery_list:
            return None, "Không dự đoán được font.", None, ocr_text
        table_str = "| # | Font | Độ tin cậy |\n|--|------|------------|\n"
        for i, r in enumerate(first_results):
            table_str += f"| #{i+1} | {r.font_name} | {r.confidence:.2f}% |\n"
        zip_path = None
        if font_names_hdf5 and font_files_hdf5 and first_results:
            zip_bytes, zip_name = create_zip_in_memory(
                first_results, font_names_hdf5, font_files_hdf5
            )
            zip_path = os.path.join(tempfile.gettempdir(), zip_name)
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
        gallery_value = gallery_list if gallery_list else None
        return gallery_value, table_str, zip_path, ocr_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Lỗi: {str(e)}", None, ""


def build_ui():
    with gr.Blocks(
        title="Font Classification + EasyOCR",
        theme=gr.themes.Soft(primary_hue="slate"),
        css="footer {visibility: hidden}",
    ) as demo:
        gr.Markdown(
            """
            # Font Classification + EasyOCR
            Tải ảnh có chữ → **EasyOCR** tìm từng vùng chữ → mỗi vùng: crop + dự đoán **Top 3 font** và hiển thị bảng chữ cái. Đầu ra = **N ảnh** (N = số vùng chữ).
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="Ảnh đầu vào",
                    type="numpy",
                    height=320,
                )
                run_btn = gr.Button("Dự đoán font + OCR", variant="primary")
            with gr.Column(scale=1):
                # Gallery: mỗi ảnh = 1 vùng chữ OCR + Top 3 font; bấm vào ảnh để xem phóng to
                image_out = gr.Gallery(
                    label="Kết quả: mỗi ảnh = vùng chữ OCR tìm thấy + Top 3 bảng chữ cái — bấm vào ảnh để xem phóng to",
                    height=360,
                    columns=1,
                    object_fit="contain",
                    show_label=True,
                )
                table_out = gr.Markdown(label="Top 3 font")
                file_out = gr.File(label="Tải ZIP 3 font")
        ocr_out = gr.Textbox(
            label="EasyOCR – Chữ nhận dạng được",
            lines=6,
            placeholder="Kết quả OCR sẽ hiển thị ở đây (mỗi dòng: nội dung + độ tin cậy).",
        )
        run_btn.click(
            fn=predict,
            inputs=[image_in],
            outputs=[image_out, table_out, file_out, ocr_out],
        )
        gr.Markdown(
            """
            ---
            **Hướng dẫn:** Chọn ảnh chứa chữ (screenshot, quảng cáo, logo…). EasyOCR trích xuất text (EN/VI), mô hình ResNet-50 dự đoán font.
            """
        )
    return demo


if __name__ == "__main__":
    load_resources()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",  # Cho phép truy cập từ mạng (public)
        server_port=7860,
        share=False,  # Đổi True để lấy link Gradio public (share=True)
    )
