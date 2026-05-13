# Font Classification – Cách chạy Inference

Hướng dẫn chạy dự đoán font từ ảnh (inference) trong project.

---

## Yêu cầu

- Python 3 (khuyến nghị 3.10+)
- PyTorch, torchvision (đã cài trong môi trường train)
- Các package trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

Trong `requirements.txt` có: `pillow`, `fonttools`, `gradio`, `easyocr`. API cần thêm: `fastapi`, `uvicorn` (nếu chưa có).

---

## Chuẩn bị trước khi chạy

1. **Checkpoint model:** Đặt file `.pth` (ví dụ `save_2/deepfont_resnet50_model_BEST.pth`) đúng đường dẫn ghi trong từng script (`MODEL_PATH`).
2. **Thư mục font:** Có thư mục chứa font theo từng family (mỗi folder = một font, ví dụ `label` hoặc `label_1`). Đường dẫn cấu hình trong code là `FONT_DIR`.
3. **API / Gradio:** Cần file `font_files.hdf5` (tạo bằng `python build_hdf5_from_label.py` từ thư mục font đang dùng).

---

## 1. Chạy batch (thư mục ảnh) – `inference.py`

Dự đoán font cho tất cả ảnh trong một thư mục, ghi kết quả ra thư mục khác.

**Bước 1:** Chỉnh trong `inference.py` (đầu file):

- `INPUT_FOLDER`: thư mục chứa ảnh (mặc định `"a"`).
- `OUTPUT_FOLDER`: thư mục ghi kết quả (mặc định `"result_a_1"`).
- `MODEL_PATH`: đường dẫn file `.pth`.
- `FONT_DIR`: thư mục font (để vẽ bảng chữ cái).

**Bước 2:** Chạy:

```bash
python inference.py
```

**Kết quả:**

- Trong `OUTPUT_FOLDER`: mỗi ảnh có một file PNG (ảnh sau padding + Top-5 bảng chữ cái) và file `prediction_log.txt` (Top-1 font cho từng ảnh).

---

## 2. Chạy API – `api_inference.py`

Phục vụ inference qua HTTP: gửi ảnh (URL hoặc base64), nhận Top-K font, ảnh kết quả, ZIP font, và (nếu bật) kết quả OCR.

**Bước 1:** Cài thêm (nếu chưa có):

```bash
pip install fastapi uvicorn
```

**Bước 2:** Tạo `font_files.hdf5` từ thư mục font (một lần):

```bash
python build_hdf5_from_label.py
```

(Script mặc định đọc từ `label_2` và ghi `font_files.hdf5`. Có thể sửa `SOURCE_DIR` trong file cho đúng.)

**Bước 3:** Chạy server:

```bash
uvicorn api_inference:app --host 0.0.0.0 --port 8000
```

Hoặc:

```bash
python api_inference.py
```

(nếu trong file có `if __name__ == "__main__": uvicorn.run(...)`).

**Bước 4:** Gửi request:

- Mở `http://localhost:8000/docs` để xem Swagger và gọi thử.
- POST `/predict_font/` với body JSON: `{"image_url": "https://..."}` hoặc `{"image_base64": "..."}`.

**Kết quả:** JSON gồm `predictions` (Top-3 font), `ocr_result`, `result_image_base64`, `zip_base64`, `zip_filename`, v.v.

---

## 3. Chạy giao diện web (Gradio) – `app_gradio.py`

Upload ảnh trên trình duyệt, xem từng vùng chữ (OCR) và Top-3 font cho từng vùng.

**Bước 1:** Đảm bảo đã cài dependencies (trong đó có `gradio`, `easyocr`):

```bash
pip install -r requirements.txt
```

**Bước 2:** Chạy:

```bash
python app_gradio.py
```

**Bước 3:** Mở địa chỉ in ra trong terminal (mặc định `http://127.0.0.1:7860` hoặc `http://0.0.0.0:7860`).

**Cách dùng:** Chọn hoặc kéo thả ảnh → bấm nút dự đoán → xem ảnh kết quả (từng vùng chữ + Top-3 font), bảng và tải ZIP. Bấm vào ảnh để xem phóng to.

**Link public (chia sẻ):** Trong `app_gradio.py`, đổi `share=False` thành `share=True` rồi chạy lại; Gradio sẽ in ra link dạng `https://....gradio.live`.

---

## Tóm tắt lệnh chạy

| Cách dùng      | Lệnh |
|----------------|------|
| Batch (thư mục)| `python inference.py` |
| API            | `uvicorn api_inference:app --host 0.0.0.0 --port 8000` |
| Giao diện web  | `python app_gradio.py` |

Chỉnh đường dẫn model, thư mục ảnh, thư mục font trong từng file tương ứng (biến cấu hình ở đầu file).
