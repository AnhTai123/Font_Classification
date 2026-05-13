import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
import os
import sys
import numpy as np
import base64
import binascii
from urllib.request import urlopen, Request
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import zipfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import h5py
from datetime import datetime
from imutils import paths

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# ==============================
# CẤU HÌNH & TẢI TÀI NGUYÊN
# ==============================
TOP_K = 3  # Top 3 font: ảnh kết quả + ZIP chỉ 3 font
MODEL_PATH = 'save_2/deepfont_resnet50_model_BEST.pth'
FONT_DIR = 'label_1'  # Thư mục chứa cấu trúc font family (khớp với font_files.hdf5)
HDF5_PATH = 'font_files.hdf5'  # Đường dẫn file HDF5 chứa names và files
IMAGE_SHAPE = (224, 224)
SEGMENT_CHECKPOINT = "save_deeplabv3plus_mobilenet/deeplabv3_best.pth"  # DeepLabV3+ segment text

# Global variables for model, mapping, EasyOCR, segment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
index_to_label = None
num_classes = 0
font_names_hdf5 = None
font_files_hdf5 = None
ocr_reader = None
segment_model = None

# ==============================
# 1. TẢI TÀI NGUYÊN (GIỮ LẠI LOGIC CẦN THIẾT)
# ==============================
def get_font_family_label(font_path, font_base_dir=None):
    """Lấy tên gia đình font từ đường dẫn file."""
    parent_dir = os.path.abspath(os.path.dirname(font_path))
    parent_dir_name = os.path.basename(parent_dir)
    
    # Nếu có font_base_dir, kiểm tra xem file có ở root không
    if font_base_dir:
        base_dir_path = os.path.abspath(font_base_dir)
        base_dir_name = os.path.basename(base_dir_path)
        
        # Nếu parent_dir_name trùng với base_dir_name, thì file ở root của font directory
        if parent_dir_name == base_dir_name:
            # File ở root, dùng tên file (loại bỏ extension) làm label
            return os.path.splitext(os.path.basename(font_path))[0]
    
    # File trong subfolder, dùng tên folder làm label
    return parent_dir_name

def build_class_mapping(font_dir):
    """
    Quét thư mục font để tạo lại ánh xạ (mapping) giống lúc train.
    Sử dụng tên folder (font family) làm label.
    """
    print(f"Đang quét {font_dir} để tạo lại ánh xạ class...")
    font_paths = list(paths.list_files(font_dir, validExts=(".ttf", ".TTF", ".otf", ".OTF")))
    if not font_paths:
        raise FileNotFoundError(f"Không tìm thấy file font nào trong: {font_dir}")
    
    all_font_families = set(get_font_family_label(p, font_dir) for p in font_paths)
    unique_labels = sorted(list(all_font_families))
    
    index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Tạo ánh xạ thành công. Tổng số {num_classes} font families (classes).")
    return index_to_label, num_classes

def load_font_mapping(h5_file_path):
    """Đọc dữ liệu font từ file HDF5 và trả về names và files đã decode."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            names_data = f['names'][:]
            files_data = f['files'][:]
        
        print(f" Đọc HDF5: {len(names_data)} entries")
        print(f"   names dtype: {names_data.dtype}, shape: {names_data.shape}")
        print(f"   files dtype: {files_data.dtype}, shape: {files_data.shape}")
        
        # Decode bytes thành string nếu cần
        names = []
        files = []
        
        for name in names_data:
            decoded_name = None
            if isinstance(name, bytes):
                decoded_name = name.decode('utf-8')
            elif isinstance(name, np.ndarray):
                # Nếu là numpy array của bytes
                if name.dtype.kind == 'S':  # String type
                    decoded_name = name.tobytes().decode('utf-8').rstrip('\x00')
                else:
                    decoded_name = str(name)
            else:
                decoded_name = str(name)
            
            if decoded_name:
                names.append(decoded_name.strip())
        
        for file_path in files_data:
            decoded_path = None
            if isinstance(file_path, bytes):
                decoded_path = file_path.decode('utf-8')
            elif isinstance(file_path, np.ndarray):
                if file_path.dtype.kind == 'S':  # String type
                    decoded_path = file_path.tobytes().decode('utf-8').rstrip('\x00')
                else:
                    decoded_path = str(file_path)
            else:
                decoded_path = str(file_path)
            
            if decoded_path:
                files.append(decoded_path.strip())
        
        print(f"[OK] Đã decode {len(names)} names và {len(files)} files từ HDF5")
        if len(names) > 0:
            print(f"   Mẫu 5 names đầu: {[n[:40] for n in names[:5]]}")
            print(f"   Mẫu 5 files đầu: {[f[:60] for f in files[:5]]}")
        
        return names, files
    except Exception as e:
        print(f"[ERR] Lỗi khi đọc file HDF5: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_font_file_path(font_name, names, files):
    """
    Tìm đường dẫn file font từ font_name trong danh sách HDF5.
    
    Args:
        font_name: Tên font từ model prediction (tên folder)
        names: Danh sách tên folder từ HDF5
        files: Danh sách tên file tương ứng từ HDF5
    
    Returns:
        Đường dẫn đầy đủ đến file font, hoặc None nếu không tìm thấy
    """
    if names is None or files is None:
        print(f"[!]  HDF5 names/files is None, không thể tìm font: {font_name}")
        return None
    
    if len(names) != len(files):
        print(f"[!]  HDF5 names và files không cùng độ dài: {len(names)} vs {len(files)}")
        return None
    
    # Normalize font name để so sánh (loại bỏ spaces, special chars)
    def normalize_name(name):
        if not name:
            return ""
        # Chuyển về lowercase, loại bỏ spaces, underscores, hyphens
        name = str(name).lower().strip()
        name = name.replace(' ', '').replace('_', '').replace('-', '')
        return name
    
    font_name_normalized = normalize_name(font_name)
    
    # Debug: In một vài tên font đầu tiên để kiểm tra
    if len(names) > 0:
        print(f" Tìm font: '{font_name}' (normalized: '{font_name_normalized}')")
        print(f"   HDF5 có {len(names)} fonts. Mẫu 5 tên đầu: {[str(n)[:30] for n in names[:5]]}")
        print(f"   Mẫu 5 file đầu: {[str(f)[:30] for f in files[:5]]}")
    
    # Tìm chính xác (exact match)
    for idx, name in enumerate(names):
        name_str = str(name).strip()
        if name_str == font_name:
            font_filename = str(files[idx]).strip()
            # Tạo đường dẫn đầy đủ: FONT_DIR/{folder_name}/{file_name}
            font_path = os.path.join(FONT_DIR, name_str, font_filename)
            print(f"[OK] Tìm thấy exact match: '{name_str}' -> '{font_path}'")
            if os.path.exists(font_path):
                return font_path
            # Thử với đường dẫn tương đối khác
            alt_paths = [
                os.path.join(name_str, font_filename),  # Relative từ root
                font_filename,  # Chỉ tên file
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"[OK] Tìm thấy ở đường dẫn thay thế: '{alt_path}'")
                    return alt_path
    
    # Tìm không phân biệt hoa thường (case-insensitive)
    font_name_lower = font_name.lower().strip()
    for idx, name in enumerate(names):
        name_str = str(name).strip()
        if name_str.lower() == font_name_lower:
            font_filename = str(files[idx]).strip()
            font_path = os.path.join(FONT_DIR, name_str, font_filename)
            print(f"[OK] Tìm thấy case-insensitive match: '{name_str}' -> '{font_path}'")
            if os.path.exists(font_path):
                return font_path
            alt_paths = [
                os.path.join(name_str, font_filename),
                font_filename,
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    return alt_path
    
    # Tìm với normalize (loại bỏ spaces, special chars)
    for idx, name in enumerate(names):
        name_normalized = normalize_name(name)
        if name_normalized and name_normalized == font_name_normalized:
            font_filename = str(files[idx]).strip()
            font_path = os.path.join(FONT_DIR, str(name).strip(), font_filename)
            print(f"[OK] Tìm thấy normalized match: '{name}' -> '{font_path}'")
            if os.path.exists(font_path):
                return font_path
            alt_paths = [
                os.path.join(str(name).strip(), font_filename),
                font_filename,
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    return alt_path
    
    # Tìm partial match (tên font chứa trong hoặc chứa tên font cần tìm)
    for idx, name in enumerate(names):
        name_str = str(name).strip().lower()
        font_name_check = font_name.lower().strip()
        if font_name_check in name_str or name_str in font_name_check:
            font_filename = str(files[idx]).strip()
            font_path = os.path.join(FONT_DIR, str(name).strip(), font_filename)
            print(f"[!]  Tìm thấy partial match: '{name}' -> '{font_path}' (có thể không chính xác)")
            if os.path.exists(font_path):
                return font_path
            alt_paths = [
                os.path.join(str(name).strip(), font_filename),
                font_filename,
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    return alt_path
    
    print(f"[ERR] Không tìm thấy font '{font_name}' trong HDF5 ({len(names)} fonts)")
    return None


def _get_ocr_reader():
    """Lazy load EasyOCR reader (en + vi)."""
    global ocr_reader
    if ocr_reader is not None:
        return ocr_reader
    if not EASYOCR_AVAILABLE:
        return None
    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(["en", "vi"], gpu=use_gpu, verbose=False)
    return ocr_reader


def run_ocr_on_image(original_rgb):
    """
    Chạy EasyOCR trên ảnh (numpy RGB, HWC uint8). Gọi TRƯỚC khi predict font.
    Returns: list[dict] với keys "text", "confidence"; hoặc [] nếu lỗi/không cài EasyOCR.
    """
    if not EASYOCR_AVAILABLE or original_rgb is None or original_rgb.size == 0:
        return []
    try:
        reader = _get_ocr_reader()
        if reader is None:
            return []
        img_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        raw = reader.readtext(img_bgr)
        out = []
        for (_bbox, text, conf) in raw:
            out.append({"text": text.strip(), "confidence": round(float(conf), 4)})
        return out
    except Exception as e:
        print(f"[!]  EasyOCR lỗi: {e}")
        return []


def run_ocr_on_image_with_boxes(original_rgb):
    """
    Chạy EasyOCR, trả về mỗi vùng chữ kèm bbox để crop.
    Returns: list[dict] với keys "bbox", "text", "confidence".
    bbox = (x_min, y_min, x_max, y_max) (integer) để crop ảnh.
    """
    if not EASYOCR_AVAILABLE or original_rgb is None or original_rgb.size == 0:
        return []
    try:
        reader = _get_ocr_reader()
        if reader is None:
            return []
        img_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        raw = reader.readtext(img_bgr)
        out = []
        for (bbox_4pts, text, conf) in raw:
            xs = [p[0] for p in bbox_4pts]
            ys = [p[1] for p in bbox_4pts]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            out.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "text": text.strip(),
                "confidence": round(float(conf), 4),
            })
        return out
    except Exception as e:
        print(f"[!]  EasyOCR lỗi: {e}")
        return []


# --- Segment (DeepLabV3+): checkpoint từ train_deeplabv3 (smp 1 kênh, sigmoid) ---
SEGMENT_IMAGE_SIZE = 224
SEGMENT_MEAN = (0.485, 0.456, 0.406)
SEGMENT_STD = (0.229, 0.224, 0.225)
BRIGHTEN_ALPHA = 1.35
BRIGHTEN_BETA = 25
SEGMENT_THRESHOLD = 0.5

try:
    import train_deeplabv3 as _t3
    _TRAIN_DEEPLAB_AVAILABLE = True
except ImportError:
    _TRAIN_DEEPLAB_AVAILABLE = False
    _t3 = None


def pad_image_to_224x224_segment(img_rgb, target_size=224, pad_value=0):
    """
    Padding letterbox giống inference_deeplabv3: giữ tỉ lệ, phần thiếu = pad_value (mặc định 0 = đen).
    img_rgb: [H,W,3] RGB uint8. Trả về [target_size, target_size, 3] uint8.
    """
    h, w = img_rgb.shape[:2]
    if h == target_size and w == target_size:
        return img_rgb
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _get_segment_model():
    """
    Lazy load DeepLabV3+ từ checkpoint train_deeplabv3 (smp, 1 kênh output, sigmoid).
    Cùng cách load với inference_deeplabv3.py.
    """
    global segment_model
    if segment_model is not None:
        return segment_model
    if not os.path.exists(SEGMENT_CHECKPOINT):
        print(f"[!]  Segment checkpoint không tồn tại: {SEGMENT_CHECKPOINT}")
        return None
    if not _TRAIN_DEEPLAB_AVAILABLE or _t3 is None:
        print("[!]  Cần import train_deeplabv3 để load segment (smp DeepLabV3+).")
        return None
    try:
        checkpoint = torch.load(SEGMENT_CHECKPOINT, map_location=device)
        saved_args = checkpoint.get("args") or {}
        backbone = saved_args.get("backbone", "mobilenet")
        model_type = saved_args.get("model_type", "deeplabv3plus")
        model_state = checkpoint.get("model")
        if model_state is None:
            print("[!]  Checkpoint không có key 'model'.")
            return None
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        segment_model = _t3.build_deeplabv3(
            num_output_channels=1,
            pretrained=False,
            backbone=backbone,
            model_type=model_type,
        )
        segment_model.load_state_dict(model_state, strict=True)
        segment_model = segment_model.to(device).eval()
        print(f"[OK] Segment model đã tải: {SEGMENT_CHECKPOINT} (backbone={backbone}, model_type={model_type})")
    except Exception as e:
        print(f"[!]  Lỗi tải segment model: {e}")
        import traceback
        traceback.print_exc()
        segment_model = None
    return segment_model


def run_segment(image_224_rgb):
    """
    Chạy segment trên ảnh 224x224 RGB (đã pad letterbox). Trả về mask (224,224) 0/255 uint8 (255 = text).
    Giống inference_deeplabv3: normalize ImageNet → model (1 kênh) → sigmoid → threshold.
    """
    seg = _get_segment_model()
    if seg is None:
        return None
    try:
        img = image_224_rgb.astype(np.float32) / 255.0
        img = (img - np.array(SEGMENT_MEAN)) / np.array(SEGMENT_STD)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = seg(x)
        logits = out["out"] if isinstance(out, dict) else out
        logits = logits[:, 0:1]  # [1, 1, H, W]
        if logits.shape[-2:] != (SEGMENT_IMAGE_SIZE, SEGMENT_IMAGE_SIZE):
            logits = F.interpolate(
                logits, size=(SEGMENT_IMAGE_SIZE, SEGMENT_IMAGE_SIZE),
                mode="bilinear", align_corners=False
            )
        prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        mask = (prob > SEGMENT_THRESHOLD).astype(np.uint8) * 255
        return mask
    except Exception as e:
        print(f"[!]  Segment inference lỗi: {e}")
        return None


def apply_mask_and_brighten(image_rgb, mask_224):
    """
    Giống inference_deeplabv3: nhân mask với ảnh đầu vào; vùng chữ giữ màu rồi kéo sáng, background giữ đen (0).
    image_rgb: (224,224,3) uint8; mask_224: (224,224) 0/255 uint8.
    Trả về (224,224,3) uint8 — đây là ảnh đưa vào classification.
    """
    if mask_224 is None:
        return image_rgb
    mask_float = (mask_224 > 127).astype(np.float32)
    mask_3 = np.expand_dims(mask_float, axis=2)
    result = (image_rgb.astype(np.float32) * mask_3).astype(np.float32)
    text_region = mask_3[:, :, 0] > 0.5
    result[text_region] = np.clip(
        result[text_region] * BRIGHTEN_ALPHA + BRIGHTEN_BETA, 0, 255
    )
    return result.astype(np.uint8)


def segment_then_classify_image(crop_rgb):
    """
    Pipeline: pad letterbox 224 (nền trắng, sau OCR) → segment → nhân mask với ảnh (nền đen) → kéo sáng vùng chữ.
    Ảnh sau bước đó (224x224) là output và đưa vào classification.
    """
    padded = pad_image_to_224x224_segment(crop_rgb, target_size=SEGMENT_IMAGE_SIZE, pad_value=255)
    mask_224 = run_segment(padded)
    if mask_224 is not None:
        return apply_mask_and_brighten(padded, mask_224)
    return padded


def define_model_architecture(num_classes):
    """
    Cấu trúc giống hệt inference.py (và train_1.py): ResNet50 + fc
    (Linear→ReLU→BatchNorm1d→Dropout→Linear→ReLU→BatchNorm1d→Linear).
    weights=None vì sẽ load từ checkpoint.
    """
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, num_classes),
    )
    return model

def load_trained_model(model_path, num_classes_from_mapping, device):
    """
    Load model và mapping từ checkpoint.
    Validate num_classes khớp với checkpoint.
    """
    print("Đang tải model...")
    try:
        checkpoint_data = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
            checkpoint_num_classes = checkpoint_data.get('num_classes')
            checkpoint_index_to_label = checkpoint_data.get('index_to_label')
            checkpoint_label_to_index = checkpoint_data.get('label_to_index')
        else:
            state_dict = checkpoint_data
            checkpoint_num_classes = None
            checkpoint_index_to_label = None
            checkpoint_label_to_index = None
            print("[!]  WARNING: Checkpoint không có mapping. Đây là checkpoint cũ.")
        
        if checkpoint_num_classes is not None:
            if checkpoint_num_classes != num_classes_from_mapping:
                print("[ERR] LỖI NGHIÊM TRỌNG: num_classes không khớp!")
                print(f"   Checkpoint: {checkpoint_num_classes} classes")
                print(f"   Mapping hiện tại: {num_classes_from_mapping} classes")
                raise RuntimeError("num_classes mismatch between checkpoint and current mapping")
            print(f"OK num_classes khớp: {checkpoint_num_classes} classes")
        
        model_num_classes = checkpoint_num_classes if checkpoint_num_classes is not None else num_classes_from_mapping
        model = define_model_architecture(model_num_classes)
        
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[new_key] = v

        # Checkpoint cũ (train_1): fc là Sequential nên có fc.0, fc.1, ... Map fc.1 -> fc nếu cần.
        if 'fc.1.weight' in cleaned_state_dict and 'fc.weight' not in cleaned_state_dict:
            cleaned_state_dict['fc.weight'] = cleaned_state_dict.pop('fc.1.weight')
            cleaned_state_dict['fc.bias'] = cleaned_state_dict.pop('fc.1.bias')

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)

        if missing_keys:
            print(f"[!]  WARNING: Thiếu keys trong checkpoint (sẽ dùng random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                print(f"   Missing keys: {missing_keys}")
            else:
                print(f"   Missing keys (first 10): {list(missing_keys)[:10]}...")

        if unexpected_keys:
            print(f"[!]  WARNING: Keys không mong đợi trong checkpoint (sẽ bỏ qua): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected keys: {unexpected_keys}")
            else:
                print(f"   Unexpected keys (first 10): {list(unexpected_keys)[:10]}...")

        print("OK Đã load model state thành công (strict=False)")
    except RuntimeError as e:
        print(f"[ERR] LỖI: Không tải được model từ {model_path}. Chi tiết: {e}")
        raise
    except Exception as e:
        print(f"[ERR] LỖI: Không tải được model từ {model_path}")
        print(f"   Chi tiết: {e}")
        raise
    
    model = model.to(device).float()
    model.eval()
    print("Model sẵn sàng.")
    return model, checkpoint_index_to_label, checkpoint_label_to_index

# ==============================
# 2. PREPROCESSING & TRANSFORM
# ==============================
def ensure_size(image, **kwargs):
    """Đảm bảo kích thước ảnh 224x224"""
    target_h, target_w = IMAGE_SHAPE
    img = image.copy()
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    temp = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    temp[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return temp

class EnsureSizeTransform(ImageOnlyTransform):
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        return ensure_size(image)

val_transform = A.Compose([
    EnsureSizeTransform(p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image_from_bytes(image_bytes, transform):
    """Xử lý ảnh từ bytes. Trả về (image_tensor, original_rgb) hoặc (None, None)."""
    try:
        if image_bytes is None or len(image_bytes) == 0:
            print("[ERR] image_bytes rỗng hoặc None")
            return None, None
        
        # Kiểm tra magic bytes để xác định định dạng ảnh
        if len(image_bytes) < 4:
            print(f"[ERR] Bytes quá ngắn: {len(image_bytes)}")
            return None, None
            
        print(f" Magic bytes: {image_bytes[:4].hex()}")
        
        # Thử decode bằng cv2 trước
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        used_pil = False
        
        # Nếu cv2 thất bại, thử dùng PIL
        if image is None:
            print("[!]  cv2.imdecode thất bại, thử dùng PIL Image...")
            try:
                # Mở ảnh bằng PIL
                pil_image = Image.open(BytesIO(image_bytes))
                # Convert PIL Image sang RGB numpy array
                image = np.array(pil_image.convert('RGB'))
                used_pil = True
                print(f"[OK] PIL Image decode thành công. Shape: {image.shape}, Mode: {pil_image.mode}")
            except Exception as pil_error:
                print(f"[ERR] PIL Image cũng thất bại: {pil_error}")
                import traceback
                traceback.print_exc()
                print(f"[ERR] cv2.imdecode trả về None. Kích thước bytes: {len(image_bytes)}")
                print("Có thể file ảnh không hợp lệ hoặc định dạng không được hỗ trợ.")
                return None, None
            
        # Kiểm tra số kênh màu
        if len(image.shape) == 2:
            # Ảnh grayscale, chuyển sang RGB (nếu dùng PIL thì đã RGB rồi)
            if not used_pil:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # PIL: grayscale -> RGB
                image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # Ảnh RGBA
                if not used_pil:
                    # cv2: RGBA -> BGR -> RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # PIL đã convert sang RGB rồi, không cần làm gì
                    pass
            elif image.shape[2] == 3:
                # Ảnh 3 kênh
                if not used_pil:
                    # cv2: BGR -> RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # PIL đã là RGB rồi, không cần làm gì
                    pass
            else:
                print(f"[ERR] Định dạng ảnh không hỗ trợ. Shape: {image.shape}")
                return None, None
        else:
            print(f"[ERR] Định dạng ảnh không hỗ trợ. Shape: {image.shape}")
            return None, None
        
        # Đảm bảo image là RGB và có shape (H, W, 3)
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"[ERR] Image shape không đúng sau khi xử lý: {image.shape}")
            return None, None
        
        # Đảm bảo dtype là uint8 và values trong range [0, 255]
        if image.dtype != np.uint8:
            print(f"[!]  Image dtype không phải uint8: {image.dtype}, đang convert...")
            if image.max() <= 1.0:
                # Values đã normalize [0, 1], convert về [0, 255]
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Đảm bảo values trong range [0, 255]
        if image.max() > 255 or image.min() < 0:
            print(f"[!]  Image values ngoài range [0, 255]: min={image.min()}, max={image.max()}, đang clip...")
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        print(f"[OK] Image sau khi xử lý. Shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
        
        # Ảnh gốc (không padding) để ghép vào ảnh kết quả
        original_rgb = image.copy()
        
        # Áp dụng transform (resize/pad 224x224) cho model
        try:
            augmented = transform(image=image)
            image_tensor = augmented['image']
            image_tensor = image_tensor.unsqueeze(0)
            
            print(f"[OK] Xử lý ảnh thành công. Shape tensor: {image_tensor.shape}")
            return image_tensor, original_rgb
        except Exception as transform_error:
            print(f"[ERR] Lỗi khi áp dụng transform: {transform_error}")
            import traceback
            traceback.print_exc()
            return None, None
        
    except Exception as e:
        print(f"[ERR] Lỗi khi xử lý ảnh: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==============================
# 3. FASTAPI SETUP
# ==============================

app = FastAPI(
    title="DeepFont ResNet-50 Inference API",
    description="API dự đoán font chữ từ ảnh đầu vào (URL hoặc Base64)."
)

# Không cần tạo thư mục api_results nữa vì file ZIP được tạo trong memory

# ==============================
# PYDANTIC MODELS (phải định nghĩa trước các endpoint)
# ==============================
class InferenceRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    
    class Config:
        # Cho phép các trường None
        schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "image_base64": None
            }
        }

class Prediction(BaseModel):
    font_name: str
    confidence: float

class OcrItem(BaseModel):
    text: str
    confidence: float


class InferenceResponse(BaseModel):
    top_k: int = TOP_K
    ocr_result: list[OcrItem] = []  # EasyOCR chạy trước, danh sách text nhận dạng được
    predictions: list[Prediction]
    result_image_base64: str | None = None
    zip_base64: str | None = None
    zip_filename: str | None = None
    status: str = "success"

@app.get("/")
def root():
    """Endpoint kiểm tra API sống."""
    return {
        "status": "ok",
        "message": "Font inference API is running",
        "docs": "/docs",
        "predict_endpoint": "/predict_font/",
    }

@app.get("/health")
def health_check():
    """Kiểm tra trạng thái model và các tài nguyên."""
    global model, index_to_label, font_names_hdf5, font_files_hdf5
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "mapping_loaded": index_to_label is not None,
        "num_classes": len(index_to_label) if index_to_label else 0,
        "hdf5_loaded": font_names_hdf5 is not None and font_files_hdf5 is not None,
        "hdf5_count": len(font_names_hdf5) if font_names_hdf5 is not None else 0,
    }

@app.post("/test_image_decode")
async def test_image_decode(request: InferenceRequest):
    """Endpoint test để debug việc decode ảnh."""
    import base64
    
    debug_info = {
        "has_image_url": request.image_url is not None,
        "has_image_base64": request.image_base64 is not None,
    }
    
    if request.image_base64:
        try:
            # Xử lý Base64
            if ';base64,' in request.image_base64:
                encoded_data = request.image_base64.split(';base64,')[-1]
            else:
                encoded_data = request.image_base64.strip()
            
            # Decode
            image_bytes = base64.b64decode(encoded_data, validate=True)
            debug_info["base64_decoded"] = True
            debug_info["image_bytes_length"] = len(image_bytes)
            debug_info["magic_bytes"] = image_bytes[:8].hex() if len(image_bytes) >= 8 else None
            
            # Test cv2 decode
            import cv2
            import numpy as np
            image_np = np.frombuffer(image_bytes, np.uint8)
            image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            debug_info["cv2_decode_success"] = image_cv2 is not None
            if image_cv2 is not None:
                debug_info["cv2_shape"] = list(image_cv2.shape)
                debug_info["cv2_dtype"] = str(image_cv2.dtype)
            
            # Test PIL decode
            from PIL import Image
            from io import BytesIO
            try:
                pil_image = Image.open(BytesIO(image_bytes))
                pil_array = np.array(pil_image.convert('RGB'))
                debug_info["pil_decode_success"] = True
                debug_info["pil_shape"] = list(pil_array.shape)
                debug_info["pil_format"] = pil_image.format
                debug_info["pil_mode"] = pil_image.mode
            except Exception as pil_err:
                debug_info["pil_decode_success"] = False
                debug_info["pil_error"] = str(pil_err)
            
            # Test transform
            try:
                if image_cv2 is not None:
                    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
                    augmented = val_transform(image=image_rgb)
                    tensor = augmented['image']
                    debug_info["transform_success"] = True
                    debug_info["tensor_shape"] = list(tensor.shape)
                else:
                    debug_info["transform_success"] = False
                    debug_info["transform_error"] = "cv2 decode failed"
            except Exception as transform_err:
                debug_info["transform_success"] = False
                debug_info["transform_error"] = str(transform_err)
            
        except Exception as err:
            debug_info["error"] = str(err)
            import traceback
            debug_info["traceback"] = traceback.format_exc()
    
    return debug_info

CHARS_52 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
GRID_COLS, GRID_ROWS = 13, 4

def create_alphabet_grid(font_name, names, files, width=700, height=1000):
    """Tạo bảng 52 chữ A-Z, a-z cho font. Trả về numpy RGB (H,W,3) uint8."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font_path = get_font_file_path(font_name, names, files) if (names and files) else None
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    default_font = ImageFont.load_default()
    cell_w, cell_h = width // GRID_COLS, height // GRID_ROWS
    base_size = max(36, min(cell_w, cell_h) - 4)
    try:
        load_font = ImageFont.truetype(font_path, base_size) if font_path else None
    except Exception:
        load_font = None
    if load_font is None:
        draw.text((width // 2 - 80, height // 2 - 20), "Không load được font", fill="#c0392b", font=default_font)
        return np.array(pil_img)
    start_y = 8
    for idx, char in enumerate(CHARS_52):
        col, row = idx % GRID_COLS, idx // GRID_COLS
        cx = col * cell_w + cell_w // 2
        cy = start_y + row * cell_h + cell_h // 2
        try:
            bbox = draw.textbbox((0, 0), char, font=load_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 30, 30
        scale = min(0.9 * cell_w / max(tw, 1), 0.9 * cell_h / max(th, 1))
        if scale < 1:
            new_size = max(24, int(base_size * scale))
            try:
                load_font = ImageFont.truetype(font_path, new_size)
            except Exception:
                pass
            try:
                bbox = draw.textbbox((0, 0), char, font=load_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = 30, 30
        x, y = cx - tw // 2, cy - th // 2
        x = max(2, min(x, width - tw - 2))
        y = max(2, min(y, height - th - 2))
        try:
            draw.text((x, y), char, fill="#000000", font=load_font)
        except Exception:
            draw.text((x, y), char, fill="#000000", font=default_font)
    return np.array(pil_img)

def create_result_image(original_rgb, results_top3, names, files):
    """
    Tạo ảnh kết quả: ảnh gốc (không padding) + 3 bảng chữ cái Top 3. Trả về PNG bytes.
    """
    if original_rgb is None or not results_top3:
        return None
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.3, 1, 1, 1], hspace=0.3, wspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(original_rgb)
    ax0.set_title(f"Ảnh đầu vào\n{original_rgb.shape[1]}×{original_rgb.shape[0]}", fontsize=14, fontweight="bold", pad=12, color="#2c3e50")
    ax0.axis("off")
    for i in range(3):
        ax = fig.add_subplot(gs[i + 1])
        ax.axis("off")
        if i < len(results_top3):
            r = results_top3[i]
            alphabet = create_alphabet_grid(r.font_name, names, files, width=700, height=1000)
            ax.imshow(alphabet)
            ax.set_title(f"#{i + 1}: {r.font_name}\n({r.confidence:.2f}%)", fontsize=12, fontweight="bold", pad=10, color="#2c3e50", bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", edgecolor="#34495e", linewidth=1.5, alpha=0.95))
        else:
            ax.set_facecolor("#ecf0f1")
    plt.suptitle("Kết quả dự đoán font - Top 3", fontsize=18, fontweight="bold", y=0.98, color="#2c3e50")
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def create_zip_in_memory(top_k_results, names, files):
    """
    Tạo file ZIP trong memory (không lưu trên server) chứa 3 file font và file TXT.
    File TXT được thêm vào trong ZIP.
    
    Args:
        top_k_results: Danh sách kết quả dự đoán (list[Prediction])
        names: Danh sách tên font từ HDF5
        files: Danh sách đường dẫn file font từ HDF5
    
    Returns:
        Tuple (zip_bytes, zip_filename) - bytes của ZIP file và tên file
    """
    # Tạo tên file với timestamp để tránh trùng lặp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"predicted_fonts_{timestamp}.zip"
    txt_filename = f"font_scores_{timestamp}.txt"
    
    # Tạo ZIP trong memory
    zip_buffer = BytesIO()
    fonts_added = 0
    
    # Tạo nội dung file TXT trong memory
    txt_content_lines = []
    txt_content_lines.append("KẾT QUẢ DỰ ĐOÁN FONT\n")
    txt_content_lines.append("=" * 50 + "\n\n")
    
    font_files_to_add = []
    
    for i, result in enumerate(top_k_results, 1):
        font_name = result.font_name
        score = result.confidence
        font_file = get_font_file_path(font_name, names, files)
        
        # Ghi vào nội dung TXT
        line = f"#{i}: {font_name} - {score:.2f}%\n"
        txt_content_lines.append(line)
        
        if font_file and os.path.exists(font_file):
            fonts_added += 1
            font_files_to_add.append((i, font_name, font_file))
            print(f"[OK] Đã tìm thấy và thêm font file: {os.path.basename(font_file)}")
        else:
            txt_content_lines.append(f"  [!]  Cảnh báo: Không tìm thấy file font cho '{font_name}'\n")
            print(f"[ERR] Không tìm thấy file font cho: {font_name}")
    
    txt_content_lines.append(f"\nTổng số font (Top 3): {len(top_k_results)}\n")
    txt_content_lines.append(f"Font đã thêm vào ZIP: {fonts_added}\n")
    
    # Tạo file ZIP trong memory
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Thêm file TXT vào ZIP (từ string, không cần file trên disk)
        txt_content_str = ''.join(txt_content_lines)
        zipf.writestr(txt_filename, txt_content_str.encode('utf-8'))
        
        # Thêm các file font vào ZIP
        for i, font_name, font_file in font_files_to_add:
            if font_file and os.path.exists(font_file):
                # Lấy tên file font (không bao gồm đường dẫn đầy đủ)
                font_basename = os.path.basename(font_file)
                # Đổi tên file trong ZIP để dễ nhận biết (ví dụ: 1_Arial.ttf)
                zip_internal_name = f"{i}_{font_name}_{font_basename}"
                # Đọc file font và thêm vào ZIP
                with open(font_file, 'rb') as f:
                    zipf.writestr(zip_internal_name, f.read())
    
    # Lấy bytes từ buffer
    zip_bytes = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"[OK] Đã tạo file ZIP (3 font + TXT): {len(zip_bytes)} bytes")
    
    return zip_bytes, zip_filename

def predict_tensor(model, tensor, device, index_to_label, top_k=5):
    """Hàm dự đoán"""
    if tensor is None:
        return None
    tensor = tensor.to(device)
    model.eval()  # đảm bảo eval mode
    with torch.no_grad():
        out = model(tensor)
        # Nếu vì lý do nào đó model trả về tuple, luôn lấy main output
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)[0]
        top_p, top_i = torch.topk(probs, top_k)
    
    results = []
    for i, p in zip(top_i, top_p):
        results.append(Prediction(
            font_name=index_to_label[i.item()],
            confidence=round(p.item()*100, 2)
        ))
    return results

async def process_image_input(request):
    """
    Xử lý đầu vào ảnh từ URL hoặc Base64.
    """
    print(f" Nhận request - image_url: {request.image_url is not None}, image_base64: {request.image_base64 is not None}")
    
    image_bytes = None
    
    # --- Xử lý đầu vào từ URL ---
    if request.image_url:
        try:
            print(f" Đang tải ảnh từ URL: {request.image_url[:50]}...")
            req = Request(request.image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                image_bytes = response.read()
            print(f"[OK] Đã tải ảnh từ URL, kích thước: {len(image_bytes)} bytes")
        except Exception as e:
            error_msg = f"Lỗi khi tải ảnh từ URL: {str(e)}"
            print(f"[ERR] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
            
    # --- Xử lý đầu vào từ Base64 ---
    elif request.image_base64:
        try:
            print(f" Đang xử lý Base64, độ dài: {len(request.image_base64)} ký tự")
            # Xử lý cả Base64 có prefix data:image/...;base64, và Base64 thuần
            if ';base64,' in request.image_base64:
                # Có prefix data URI
                encoded_data = request.image_base64.split(';base64,')[-1]
                print("Có prefix data URI")
            else:
                # Base64 thuần, không có prefix
                encoded_data = request.image_base64.strip()
                print("Base64 thuần, không có prefix")
            
            # Decode Base64
            image_bytes = base64.b64decode(encoded_data, validate=True)
            print(f"[OK] Đã decode Base64, kích thước: {len(image_bytes)} bytes")
        except (base64.binascii.Error, binascii.Error) as e:
            error_msg = f"Lỗi: Base64 không hợp lệ. Chi tiết: {str(e)}"
            print(f"[ERR] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Lỗi khi giải mã Base64: {str(e)}"
            print(f"[ERR] {error_msg}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=error_msg)
    else:
        # Không có cả image_url và image_base64
        error_msg = "Vui lòng cung cấp 'image_url' hoặc 'image_base64'."
        print(f"[ERR] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # --- Kiểm tra đầu vào ---
    if image_bytes is None or len(image_bytes) == 0:
        error_msg = "Vui lòng cung cấp 'image_url' hoặc 'image_base64' hợp lệ."
        print(f"[ERR] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # --- Preprocess ảnh ---
    try:
        image_tensor, original_rgb = preprocess_image_from_bytes(image_bytes, val_transform)
    except Exception as e:
        error_msg = f"Lỗi khi xử lý ảnh: {str(e)}"
        print(f"[ERR] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    if image_tensor is None:
        error_msg = "Không thể xử lý và chuyển ảnh thành tensor. Vui lòng kiểm tra định dạng ảnh (PNG, JPG, JPEG)."
        print(f"[ERR] {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    return image_tensor, original_rgb

@app.post("/predict_font/", response_model=InferenceResponse)
async def predict_font_endpoint(request: InferenceRequest):
    """
    Luồng: 1) Decode ảnh  2) EasyOCR trên ảnh gốc  3) Segment (pad 224, mask, brighten)  4) Classification.
    Trả về: ocr_result (EasyOCR), Top 3 font, ảnh kết quả (ảnh đã segment), ZIP 3 font.
    """
    global model, index_to_label, font_names_hdf5, font_files_hdf5

    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load.")
    if index_to_label is None:
        raise HTTPException(status_code=500, detail="Class mapping chưa được load.")

    try:
        _, original_rgb = await process_image_input(request)
        # 1) EasyOCR trước (trên ảnh gốc)
        ocr_raw = run_ocr_on_image(original_rgb)
        ocr_result = [OcrItem(text=x["text"], confidence=x["confidence"]) for x in ocr_raw]
        # 2) Segment trước (pad 224, segment, mask + brighten), sau đó mới classification
        image_for_clf = segment_then_classify_image(original_rgb)
        augmented = val_transform(image=image_for_clf)
        tensor_clf = augmented["image"].unsqueeze(0)
        results = predict_tensor(model, tensor_clf, device, index_to_label, TOP_K)
        if not results:
            raise HTTPException(status_code=500, detail="Lỗi dự đoán từ mô hình.")
        results = results[:TOP_K]
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    result_image_base64 = None
    try:
        result_image_bytes = create_result_image(image_for_clf, results, font_names_hdf5, font_files_hdf5)
        if result_image_bytes:
            result_image_base64 = base64.b64encode(result_image_bytes).decode("utf-8")
    except Exception as e:
        print(f"[!]  Lỗi tạo ảnh kết quả: {e}")

    zip_base64 = None
    zip_filename = None
    if font_names_hdf5 is not None and font_files_hdf5 is not None:
        try:
            zip_bytes, zip_filename = create_zip_in_memory(results, font_names_hdf5, font_files_hdf5)
            zip_base64 = base64.b64encode(zip_bytes).decode("utf-8")
        except Exception as e:
            print(f"[!]  Lỗi tạo ZIP: {e}")

    return InferenceResponse(
        ocr_result=ocr_result,
        predictions=results,
        result_image_base64=result_image_base64,
        zip_base64=zip_base64,
        zip_filename=zip_filename,
        status="success",
    )

@app.on_event("startup")
def load_resources():
    """Tải model, ánh xạ class và HDF5 mapping khi API khởi động."""
    global model, index_to_label, num_classes, font_names_hdf5, font_files_hdf5
    
    print("--- KHỞI ĐỘNG API VÀ TẢI TÀI NGUYÊN ---")
    
    # 1. Tải mapping: ưu tiên mapping trong checkpoint, nếu không có thì rebuild
    try:
        checkpoint_data = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint_data, dict) and 'index_to_label' in checkpoint_data:
            index_to_label = checkpoint_data['index_to_label']
            num_classes = checkpoint_data.get('num_classes')
            print(f"OK Đã load mapping từ checkpoint: {num_classes} classes (ưu tiên dùng mapping gốc).")
        else:
            print("[!]  Checkpoint không có mapping. Đang rebuild mapping từ FONT_DIR (có thể không khớp 100% với training).")
            index_to_label, num_classes = build_class_mapping(FONT_DIR)
    except Exception as e:
        print(f"[!]  Không thể load mapping từ checkpoint: {e}")
        print("   Đang rebuild mapping từ FONT_DIR...")
        index_to_label, num_classes = build_class_mapping(FONT_DIR)

    # 2. Tải HDF5 mapping (names và files)
    try:
        if os.path.exists(HDF5_PATH):
            font_names_hdf5, font_files_hdf5 = load_font_mapping(HDF5_PATH)
            if font_names_hdf5 is not None and font_files_hdf5 is not None:
                print(f"[OK] HDF5 mapping đã tải: {len(font_names_hdf5)} font entries.")
            else:
                print(f"[!]  Warning: Không đọc được dữ liệu từ {HDF5_PATH}")
                font_names_hdf5 = None
                font_files_hdf5 = None
        else:
            print(f"[!]  Warning: File {HDF5_PATH} không tồn tại. Chức năng tạo ZIP sẽ bị tắt.")
            font_names_hdf5 = None
            font_files_hdf5 = None
    except Exception as e:
        print(f"[!]  Warning: Lỗi khi tải HDF5 mapping: {e}")
        print("Chức năng tạo ZIP sẽ bị tắt nhưng API vẫn hoạt động.")
        font_names_hdf5 = None
        font_files_hdf5 = None

    # 3. Tải mô hình với validate num_classes
    try:
        model_loaded, ckpt_idx_to_label, _ = load_trained_model(MODEL_PATH, num_classes, device)
        if ckpt_idx_to_label is not None:
            # Nếu checkpoint cung cấp mapping, dùng mapping đó để đảm bảo khớp tuyệt đối
            index_to_label = ckpt_idx_to_label
            num_classes = len(index_to_label)
            print(f"OK Mapping được cập nhật từ checkpoint (num_classes={num_classes}).")
        model = model_loaded
        print("[OK] Mô hình đã tải và sẵn sàng.")
    except Exception as e:
        print(f"[ERR] LỖI TẢI MODEL: {e}")
        sys.exit(1)

# ==============================
# CHẠY API
# ==============================
if __name__ == "__main__":
    print("\nAPI sẵn sàng chạy trên http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
