import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import cv2
import numpy as np
import os
import pandas as pd
from imutils import paths
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import shutil
import random
import multiprocessing
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import colorsys  # For RGB <-> HSV conversion
import string
import sys
import traceback
import signal
import gc
import matplotlib.pyplot as plt
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.multiclass')
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH ---
FONT_TRAIN_DIR = 'label_1'
FONT_VAL_DIR = 'label_1'

# WORDS_FILE không còn được sử dụng - tự động generate từ ngẫu nhiên
IMAGE_SHAPE = (224, 224)
# Erode GT mask (px). 0 = tắt. Sau khi mask vẽ stroke_width=0 (khớp chữ), không cần erode nữa (erode sẽ làm mask biến mất).
ERODE_GT_MASK_PX_TRAIN = 0
ERODE_GT_MASK_PX_VAL = 0
# Ảnh đầu vào luôn 224x224
MIN_GENERATED_WIDTH = 224
MAX_GENERATED_WIDTH = 224
MIN_GENERATED_HEIGHT = 224
MAX_GENERATED_HEIGHT = 224
NUM_CLASSES = 14037
BATCH_SIZE = 128  # Giảm từ 128 xuống 64 để tiết kiệm RAM/VRAM
EPOCHS = 205
LEARNING_RATE = 0.0003
# Khi train tiếp 50k từ best checkpoint (14k→50k): giảm LR để tránh "quên" (1/2 hoặc 1/10)
LEARNING_RATE_FROM_BEST = LEARNING_RATE * 0.5  # 0.00015; có thể đổi 0.1 nếu muốn thấp hơn
# True = load từ BEST (chất lượng tốt nhất trên val), dùng khi train tiếp với nhiều class hơn (vd 50k font)
RESUME_FROM_BEST = True  # False = resume từ LAST (đúng tiến trình epoch/optimizer)
PATIENCE = 100
NUM_WORKERS = 8  # 0 = tránh worker bị kill (Aborted/SIGABRT). Có thể thử 2 nếu máy ổn định.
SAMPLES_PER_FONT = 25

SAMPLES_PER_FONT_VAL = 5

# Giới hạn số lượng patterns/backgrounds được load vào RAM để tránh bùng memory
# Giảm xuống để tiết kiệm RAM (mỗi dataset load riêng, có 3 datasets)
MAX_PATTERNS_IN_MEMORY = 300  # Tối đa 300 patterns trong RAM (giảm từ 1000)
MAX_BACKGROUNDS_IN_MEMORY = 300  # Tối đa 300 backgrounds trong RAM (giảm từ 1000)

def get_channel_a_prob_schedule(epoch, total_epochs):
    """
    Schedule động cho CHANNEL_A_PROB theo epoch:
    - Epoch 0-20%: skeleton cao (0.6-0.7) để học shape cơ bản
    - Epoch 20-60%: giảm dần về 0.3
    - Epoch 60-100%: giảm về 0.1 để fine-tune trên domain realistic
    """
    progress = epoch / total_epochs  # 0.0 đến 1.0
    
    if progress <= 0.2:  # 0-20%
        # Linear từ 0.7 xuống 0.6
        return 0.7 - (progress / 0.2) * 0.1
    elif progress <= 0.6:  # 20-60%
        # Linear từ 0.6 xuống 0.3
        return 0.6 - ((progress - 0.2) / 0.4) * 0.3
    else:  # 60-100%
        # Linear từ 0.3 xuống 0.1
        return 0.3 - ((progress - 0.6) / 0.4) * 0.2

def get_augmentation_schedule(epoch, total_epochs):
    """
    Schedule động: Rotate + Affine scale nhẹ. Augment chữ (JPEG/blur/downscale) trong apply_text_region_augmentations.
    Rotate: 15° → 8°. Affine scale: (0.94, 1.02) → (0.96, 1.01) (giảm dần).
    """
    progress = epoch / total_epochs
    if progress <= 0.5:
        alpha = progress / 0.5
        rotate_limit = 15 - alpha * 3.5  # 15 → 11.5
        affine_scale = (0.94 - alpha * 0.01, 1.02 - alpha * 0.005)  # (0.94, 1.02) → (0.93, 1.015)
    else:
        alpha = (progress - 0.5) / 0.5
        rotate_limit = 11.5 - alpha * 3.5  # 11.5 → 8
        affine_scale = (0.93 + alpha * 0.03, 1.015 - alpha * 0.005)  # (0.93, 1.015) → (0.96, 1.01)
    return {
        'rotate_limit': max(8, int(rotate_limit)),
        'affine_scale': affine_scale
    }

def get_text_generation_schedule(epoch, total_epochs):
    """
    Schedule động cho text generation parameters theo epoch:
    - Đầu: đa dạng (85% words, 15% letters, 50% chance 2 dòng cho 2 từ)
    - Cuối: tăng xác suất 2 dòng để "lộ style" nhiều hơn (80% chance 2 dòng)
    
    Returns:
        dict với text generation parameters đã được schedule
    """
    progress = epoch / total_epochs  # 0.0 đến 1.0
    
    # Xác suất split 2 dòng khi có 2 từ: tăng dần từ 50% → 80%
    multiline_prob_2words = 0.5 + progress * 0.3  # 0.5 → 0.8
    
    # Xác suất split 2 dòng cho random letters: tăng dần từ 40% → 70%
    multiline_prob_letters = 0.4 + progress * 0.3  # 0.4 → 0.7
    
    return {
        'multiline_prob_2words': multiline_prob_2words,
        'multiline_prob_letters': multiline_prob_letters
    }

def build_train_transform(aug_params):
    """
    Train transform: Rotate + Affine scale nhẹ (toàn ảnh). JPEG/blur/downscale đã áp trên patch chữ.
    """
    return A.Compose([
        EnsureSizeTransform(p=1.0),
        A.Rotate(
            limit=aug_params['rotate_limit'],
            p=0.2,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=(255, 255, 255),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.Affine(
            scale=aug_params['affine_scale'],
            translate_percent=0,
            rotate=0,
            shear=0,
            cval=(255, 255, 255),
            mode=cv2.BORDER_CONSTANT,
            p=0.2,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def apply_text_region_augmentations(image_np, mask_np, rng):
    """
    Augment chỉ lên vùng chữ (theo mask): JPEG compression patch chữ, blur nhẹ, downscale→upscale.
    image_np: RGB uint8 [H,W,3], mask_np: float32 [0,1] [H,W]. Trả về image_np đã sửa (mask giữ nguyên).
    """
    if image_np is None or mask_np is None:
        return image_np
    assert image_np.shape[:2] == mask_np.shape[:2]
    H, W = image_np.shape[:2]
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0 or len(xs) == 0:
        return image_np
    pad = 2
    y1, y2 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
    x1, x2 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
    if y2 <= y1 or x2 <= x1:
        return image_np
    patch = image_np[y1:y2, x1:x2].copy()
    ph, pw = patch.shape[:2]
    if ph < 4 or pw < 4:
        return image_np

    # 1) JPEG compression trên patch chữ
    if rng.random() < 0.35:
        quality = rng.randint(70, 94)
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if buf is not None:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None:
                patch = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    # 2) Blur rất nhẹ trên chữ (chỉ kernel 3, sigma nhỏ)
    if rng.random() < 0.25:
        patch = cv2.GaussianBlur(patch, (3, 3), 0.35)

    # 3) Downscale → Upscale patch chữ
    if rng.random() < 0.3:
        scale = rng.uniform(0.25, 0.6)
        sw, sh = max(2, int(pw * scale)), max(2, int(ph * scale))
        small = cv2.resize(patch, (sw, sh), interpolation=cv2.INTER_AREA)
        patch = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_LINEAR)

    image_np[y1:y2, x1:x2] = patch
    return image_np


save_dir = 'save_2'

# Khi file này được import (ví dụ: train_deeplabv3.py reuse dataset/pipeline),
# ta không muốn tạo folder / in log / tạo SummaryWriter ngay lập tức.
# Chỉ chạy trong main process để tránh in "Save directory" nhiều lần khi DataLoader spawn workers.
_TRAIN1_IMPORT_SAFE = os.environ.get("TRAIN1_IMPORT_SAFE", "0") == "1"
_is_main_process = multiprocessing.current_process().name == "MainProcess"
if not _TRAIN1_IMPORT_SAFE and _is_main_process:
    os.makedirs(save_dir, exist_ok=True)
    print(f" Save directory: {os.path.abspath(save_dir)}")
    writer = SummaryWriter(f'{save_dir}/runs/font_classification_synthetic')
elif not _TRAIN1_IMPORT_SAFE and not _is_main_process:
    writer = None  # Worker process: không tạo writer
else:
    writer = None  # train_1.py sẽ tự tạo writer trong main() khi chạy trực tiếp

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global tracking cho F1 bin transitions ---
# Lưu bin assignment của mỗi font ở epoch trước (class_idx -> bin_idx)
_font_bin_history = {}  # {epoch: {class_idx: bin_idx}}
_last_tracked_epoch = -1

# Màu gốc cho mỗi bin (10 bins)
_bin_base_colors = [
    '#FF6B6B',  # 0-10%: Đỏ
    '#FF8E53',  # 10-20%: Cam đỏ
    '#FFA726',  # 20-30%: Cam
    '#FFB74D',  # 30-40%: Cam vàng
    '#FFD54F',  # 40-50%: Vàng
    '#AED581',  # 50-60%: Xanh lá nhạt
    '#81C784',  # 60-70%: Xanh lá
    '#4DB6AC',  # 70-80%: Xanh ngọc
    '#4FC3F7',  # 80-90%: Xanh dương nhạt
    '#64B5F6',  # 90-100%: Xanh dương
]

# --- 2. TỰ ĐỘNG TẠO TỪ TỪ TẤT CẢ CHỮ CÁI ---
# Bao gồm tất cả chữ cái (uppercase + lowercase) = 52 ký tự
# Có thể thêm số và ký tự đặc biệt nếu muốn
# Mặc định: cho model học cả chữ cái (a-z, A-Z) và số (0-9)
TARGET_CHARS = list(string.ascii_letters + string.digits)  # 26*2 chữ + 10 số = 62 ký tự
# Tùy chọn: thêm ký tự đặc biệt nếu muốn
# TARGET_CHARS = list(string.ascii_letters + string.digits + ".,!?;:()[]{}\"'/-")

def font_supports_all_target_chars(font_path, chars=TARGET_CHARS):
    """Kiểm tra font có hỗ trợ toàn bộ kí tự mục tiêu không."""
    try:
        font = ImageFont.truetype(str(font_path), size=80)
    except Exception:
        return False
    for ch in chars:
        try:
            bbox = font.getbbox(ch)
            if bbox is None:
                return False
        except Exception:
            return False
    return True

def generate_skeleton_image(text, font_path, font_size, image_size, angle_limit=4,
                            pattern_pil=None, use_pattern_fill_prob=0.15):
    """Channel A: tạo ảnh skeleton với màu nền và màu chữ ngẫu nhiên.
    Tạo ảnh với size ngẫu nhiên trước, sau đó pad về 224x224.
    Có thể sử dụng pattern fill cho text."""
    try:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception:
            font = ImageFont.load_default()

        # Chọn màu nền từ danh sách màu cố định cho skeleton
        SKELETON_BACKGROUND_COLORS = [
            (255, 0, 0),      # Đỏ
            (0, 255, 0),      # Xanh lá
            (0, 0, 255),      # Xanh dương
            (255, 255, 0),    # Vàng
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Cam
            (128, 0, 128),    # Tím
            (0, 0, 0),        # Đen
            (255, 255, 255)   # Trắng
        ]
        bg_color = random.choice(SKELETON_BACKGROUND_COLORS)
        
        # Chọn màu chữ dựa trên màu nền cụ thể để đảm bảo contrast cao
        text_color = choose_text_color(bg_color)
        
        # Xác định màu outline: đối lập với màu chữ
        text_luminance = calculate_luminance(text_color)
        if text_luminance > 0.5:
            # Chữ sáng → outline đen
            outline_color = (0, 0, 0)
        else:
            # Chữ tối → outline trắng
            outline_color = (255, 255, 255)

        # Tạo canvas với size ngẫu nhiên (giống như ảnh có background)
        random_w = random.randint(MIN_GENERATED_WIDTH, MAX_GENERATED_WIDTH)
        random_h = random.randint(MIN_GENERATED_HEIGHT, MAX_GENERATED_HEIGHT)
        random_size = (random_w, random_h)
        
        # Tạo canvas RGBA để có thể composite với pattern
        canvas = Image.new("RGBA", random_size, bg_color + (255,))
        draw = ImageDraw.Draw(canvas)
        
        is_multiline = "\n" in text
        line_spacing = max(0, int(font_size * 0.2))
        
        # Tính toán vùng an toàn để text không bị cắt khi resize/pad
        # Margin để đảm bảo text không bị cắt khi resize
        margin = max(5, int(min(random_w, random_h) * 0.05))  # 5% margin, tối thiểu 5px
        
        # Vùng an toàn trên canvas random_size
        safe_x1 = margin
        safe_y1 = margin
        safe_x2 = random_w - margin
        safe_y2 = random_h - margin
        safe_w = safe_x2 - safe_x1
        safe_h = safe_y2 - safe_y1
        
        # Tính text bbox
        if is_multiline:
            if hasattr(draw, "multiline_textbbox"):
                bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
                (l, t, r, b) = bbox
                text_w = r - l
                text_h = b - t
            else:
                lines = text.split("\n")
                widths, heights = [], []
                for ln in lines:
                    lb, tb, rb, bb = font.getbbox(ln)
                    widths.append(rb - lb)
                    heights.append(bb - tb)
                text_w = max(widths) if widths else font_size
                text_h = sum(heights) + line_spacing * (len(heights) - 1) if heights else font_size
                l, t = 0, 0
        else:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            l, t = bbox[0], bbox[1]
        
        # Đảm bảo text nằm trong vùng an toàn
        # Nếu text quá lớn, thu nhỏ font size
        scale_w = safe_w / (text_w + 10) if text_w + 10 > safe_w else 1.0
        scale_h = safe_h / (text_h + 10) if text_h + 10 > safe_h else 1.0
        scale = min(scale_w, scale_h, 1.0)
        
        if scale < 1.0:
            # Thu nhỏ font size
            font_size = max(28, int(font_size * scale))  # Giữ font >= 28 để tránh mất chữ
            try:
                font = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                font = ImageFont.load_default()
            # Tính lại bbox với font size mới
            if is_multiline:
                if hasattr(draw, "multiline_textbbox"):
                    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
                    (l, t, r, b) = bbox
                    text_w = r - l
                    text_h = b - t
                else:
                    lines = text.split("\n")
                    widths, heights = [], []
                    for ln in lines:
                        lb, tb, rb, bb = font.getbbox(ln)
                        widths.append(rb - lb)
                        heights.append(bb - tb)
                    text_w = max(widths) if widths else font_size
                    text_h = sum(heights) + line_spacing * (len(heights) - 1) if heights else font_size
                    l, t = 0, 0
            else:
                bbox = font.getbbox(text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                l, t = bbox[0], bbox[1]
        
        # Đặt text trong vùng an toàn (center trong safe region)
        x = safe_x1 + (safe_w - text_w) / 2 - l
        y = safe_y1 + (safe_h - text_h) / 2 - t
        
        # Không viền chữ (stroke_width=0)
        # Quyết định có dùng pattern không
        use_pattern = (pattern_pil is not None) and (random.random() < use_pattern_fill_prob)
        
        if use_pattern:
            # Vẽ chữ không viền, sau đó fill pattern
            draw_text_with_outline(
                draw, text, (x, y), font, bg_color,
                is_multiline=is_multiline, line_spacing=line_spacing
            )
            
            # Resize pattern để match với random_size nếu cần
            pattern_resized = pattern_pil.copy()
            if pattern_resized.size != random_size:
                pattern_resized = pattern_resized.resize(random_size, Image.Resampling.LANCZOS)
            
            # Tạo mask và fill pattern
            mask = Image.new("L", random_size, 0)
            draw_mask = ImageDraw.Draw(mask)
            # Mask phải vẽ giống ảnh render (bao gồm conditional spacing) để tránh chữ “dính”
            draw_text_with_conditional_spacing(
                draw_mask,
                text,
                (x, y),
                font,
                255,
                is_multiline=is_multiline,
                line_spacing=line_spacing,
                stroke_width=0,
                stroke_fill=None,
            )
            pattern_with_alpha = pattern_resized.convert("RGBA")
            pattern_with_alpha.putalpha(mask)
            canvas = Image.alpha_composite(canvas, pattern_with_alpha)
        else:
            # Vẽ chữ thường không viền
            draw_text_with_outline(
                draw, text, (x, y), font, bg_color,
                is_multiline=is_multiline, line_spacing=line_spacing
            )

        # Convert về RGB trước khi rotate
        canvas = canvas.convert("RGB")
        
        if angle_limit > 0:
            angle = random.uniform(-angle_limit, angle_limit)
            canvas = canvas.rotate(angle, expand=True, fillcolor=bg_color)
            # Không resize về random_size sau rotate để tránh cắt text
            # Canvas sau rotate có thể lớn hơn random_size, sẽ được xử lý ở bước pad

        # Pad về 224x224 (giống như prepare_canvas_and_real_region)
        w, h = canvas.size
        target_size = image_size[0]  # 224
        
        # Case 1: Ảnh lớn hơn target_size → resize với aspect ratio, sau đó pad
        if w > target_size or h > target_size:
            if w > h:
                scale = target_size / w
                new_w = target_size
                new_h = int(h * scale)
            else:
                scale = target_size / h
                new_h = target_size
                new_w = int(w * scale)
            canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h
        
        # Xử lý skeleton CHỈ trên vùng có nội dung (không bao gồm padding trắng)
        # để tránh vấn đề threshold không phân biệt được khi padding trắng chiếm phần lớn ảnh
        canvas_np = np.array(canvas)  # Ảnh gốc trước khi pad
        
        # Chuyển sang grayscale để xử lý skeleton (chỉ trên vùng có nội dung)
        img_gray = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2GRAY)
        
        inv = 255 - img_gray
        _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ngẫu nhiên hóa kích thước kernel giãn nở trong khoảng 1–3 pixel
        # để mô hình thấy được nét chữ với độ dày hơi khác nhau
        kernel_size = random.randint(1, 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        strokes = cv2.dilate(thresh, kernel, iterations=1)
        strokes = cv2.GaussianBlur(strokes, (3, 3), 0)

        # Tạo skeleton: chữ trên nền (skeleton < 255 là phần chữ)
        skeleton_content = 255 - strokes

        # Tạo mask: nơi skeleton < 255 là phần chữ (stroke)
        # Normalize skeleton về 0-1: 0 = nền, 1 = chữ
        mask_content = (255 - skeleton_content).astype(np.float32) / 255.0

        # Fallback: nếu mask quá nhỏ (không thấy chữ), dùng threshold cố định
        if mask_content.max() < 0.05:
            _, thresh_fb = cv2.threshold(inv, 30, 255, cv2.THRESH_BINARY)
            strokes_fb = cv2.dilate(thresh_fb, kernel, iterations=1)
            # CRITICAL: Không dùng GaussianBlur cho mask (sẽ làm blur)
            # Chỉ dùng cho skeleton_content (ảnh), mask sẽ được tính riêng
            strokes_fb_blurred = cv2.GaussianBlur(strokes_fb, (3, 3), 0)  # Chỉ cho skeleton
            skeleton_content = 255 - strokes_fb_blurred
            # Mask tính từ strokes_fb (không blur) để giữ binary
            mask_content = (255 - strokes_fb).astype(np.float32) / 255.0
            # CRITICAL: Threshold về binary ngay sau khi tính
            mask_content = (mask_content > 0.5).astype(np.float32)  # Binary: 0 hoặc 1
        
        # Tạo skeleton RGB cho vùng có nội dung
        skeleton_content_rgb = np.zeros((skeleton_content.shape[0], skeleton_content.shape[1], 3), dtype=np.uint8)
        
        # Áp dụng màu cho vùng có nội dung:
        # - Nền: bg_color
        # - Chữ: text_color (dựa trên mask)
        for c in range(3):
            # Interpolate giữa màu nền (bg_color[c]) và màu chữ (text_color[c])
            skeleton_content_rgb[:, :, c] = (
                bg_color[c] * (1 - mask_content) + text_color[c] * mask_content
            ).astype(np.uint8)
        
        # Tạo ảnh skeleton cuối cùng 224x224 với padding trắng
        skeleton_rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        skeleton_rgb[:, :] = (255, 255, 255)  # Padding trắng
        
        # Tính toán vị trí để paste vùng có nội dung
        x_offset = (target_size - w) // 2
        y_offset = (target_size - h) // 2
        
        # Paste vùng có nội dung vào đúng vị trí
        content_y1 = max(0, y_offset)
        content_y2 = min(target_size, y_offset + h)
        content_x1 = max(0, x_offset)
        content_x2 = min(target_size, x_offset + w)
        
        # Đảm bảo kích thước khớp
        src_h, src_w = skeleton_content_rgb.shape[:2]
        dst_h = content_y2 - content_y1
        dst_w = content_x2 - content_x1
        
        if src_h != dst_h or src_w != dst_w:
            # Resize nếu cần (trường hợp edge case)
            skeleton_content_rgb = cv2.resize(skeleton_content_rgb, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        
        skeleton_rgb[content_y1:content_y2, content_x1:content_x2] = skeleton_content_rgb

        # Dùng mask_content (0-1) làm GT mask cho seg (nét chữ = 1, nền = 0)
        # IMPORTANT: Paste mask vào cùng vị trí với skeleton_content, không resize trực tiếp
        # Tạo mask_full = zeros(224, 224) rồi paste mask_content vào cùng offset
        mask_full = np.zeros((target_size, target_size), dtype=np.float32)
        
        # Resize mask_content về đúng dst_w, dst_h (giống như skeleton_content_rgb)
        mask_content_resized = mask_content.astype(np.float32)
        src_mask_h, src_mask_w = mask_content_resized.shape[:2]
        
        if src_mask_h != dst_h or src_mask_w != dst_w:
            # Resize mask_content về đúng kích thước (giống skeleton_content_rgb)
            mask_content_resized = cv2.resize(
                mask_content_resized,
                (dst_w, dst_h),
                interpolation=cv2.INTER_NEAREST  # Dùng NEAREST để giữ binary mask
            )
        
        # Paste mask_content vào cùng vị trí với skeleton_content
        mask_full[content_y1:content_y2, content_x1:content_x2] = mask_content_resized
        
        # CRITICAL: Threshold về binary để đảm bảo mask sắc nét (không có giá trị trung gian)
        # Sau resize với INTER_NEAREST, vẫn có thể có giá trị trung gian do normalize
        mask_full = (mask_full > 0.5).astype(np.float32)  # Binary: 0 hoặc 1
        
        return skeleton_rgb, mask_full
    except Exception:
        return None, None

def crop_region_pil(image_pil, target_width, target_height):
    """Crop a random region from PIL image, resizing up if needed."""
    w, h = image_pil.size
    if w <= 0 or h <= 0:
        return image_pil
    if w < target_width or h < target_height:
        scale = max(target_width / max(w, 1), target_height / max(h, 1))
        new_size = (max(1, int(w * scale) + 4), max(1, int(h * scale) + 4))
        image_pil = image_pil.resize(new_size, Image.Resampling.BICUBIC)
    w, h = image_pil.size
    target_width = min(target_width, max(1, w - 1))
    target_height = min(target_height, max(1, h - 1))
    max_x = max(w - target_width, 0)
    max_y = max(h - target_height, 0)
    x0 = random.randint(0, max_x) if max_x > 0 else 0
    y0 = random.randint(0, max_y) if max_y > 0 else 0
    return image_pil.crop((x0, y0, x0 + target_width, y0 + target_height))

# --- 3. HÀM TRỢ GIÚP ---
def adjust_saturation(rgb_color, max_saturation=0.7, min_contrast=4.5, background_color=None):
    """
    Giảm saturation của màu nếu quá cao (quá rực rỡ) để chữ không quá chói.
    Đảm bảo contrast vẫn >= min_contrast sau khi giảm saturation.
    
    Args:
        rgb_color: Tuple (R, G, B) màu gốc
        max_saturation: Saturation tối đa cho phép (default 0.7, tức 70%)
        min_contrast: Contrast tối thiểu cần đạt (default 4.5)
        background_color: Màu background để kiểm tra contrast (optional)
    
    Returns:
        Tuple (R, G, B) màu đã được điều chỉnh saturation
    """
    if background_color is None:
        # Nếu không có background, chỉ giảm saturation đơn giản
        h, s, v = rgb_to_hsv(rgb_color)
        if s > max_saturation:
            s = max_saturation
        return hsv_to_rgb((h, s, v))
    
    h, s, v = rgb_to_hsv(rgb_color)
    current_contrast = calculate_contrast_ratio(rgb_color, background_color)
    
    # Nếu saturation đã thấp hơn max_saturation và contrast đủ, không cần điều chỉnh
    if s <= max_saturation and current_contrast >= min_contrast:
        return rgb_color
    
    # Nếu saturation quá cao, giảm xuống
    if s > max_saturation:
        # Thử giảm saturation từ từ để tìm giá trị tốt nhất
        best_color = rgb_color
        best_contrast = current_contrast
        
        # Thử các giá trị saturation từ max_saturation xuống 0.3
        for test_s in np.linspace(max_saturation, max(0.3, s - 0.3), 20):
            test_hsv = (h, test_s, v)
            test_rgb = hsv_to_rgb(test_hsv)
            test_contrast = calculate_contrast_ratio(test_rgb, background_color)
            
            if test_contrast >= min_contrast and test_contrast >= best_contrast * 0.95:
                # Nếu contrast vẫn đủ và không giảm quá nhiều, dùng màu này
                best_color = test_rgb
                best_contrast = test_contrast
                # Ưu tiên saturation thấp hơn nếu contrast vẫn đủ
                if test_s <= max_saturation:
                    break
        
        # Nếu sau khi giảm saturation mà contrast vẫn đủ, dùng màu đó
        if best_contrast >= min_contrast:
            return best_color
    
    # Nếu giảm saturation làm mất contrast, có thể cần điều chỉnh value
    if current_contrast < min_contrast:
        # Điều chỉnh value để tăng contrast
        bg_lum = calculate_luminance(background_color)
        for test_s in [s, max_saturation, max_saturation * 0.8, max_saturation * 0.6]:
            for adjustment in [0.05, 0.1, 0.15]:
                if bg_lum > 0.5:
                    # Nền sáng: giảm value (làm tối hơn)
                    test_v = max(0.0, v - adjustment)
                else:
                    # Nền tối: tăng value (làm sáng hơn)
                    test_v = min(1.0, v + adjustment)
                
                test_hsv = (h, test_s, test_v)
                test_rgb = hsv_to_rgb(test_hsv)
                test_contrast = calculate_contrast_ratio(test_rgb, background_color)
                
                if test_contrast >= min_contrast:
                    return test_rgb
    
    return rgb_color

def adjust_luminance(rgb_color, target_luminance, min_contrast=4.5, background_color=None):
    """
    Điều chỉnh luminance của màu để đạt target_luminance trong khi vẫn giữ contrast >= min_contrast.
    Sử dụng binary search để tìm giá trị value (HSV) tối ưu.
    
    Args:
        rgb_color: Tuple (R, G, B) màu gốc
        target_luminance: Luminance mục tiêu (0.0-1.0)
        min_contrast: Contrast tối thiểu cần đạt (default 4.5)
        background_color: Màu background để kiểm tra contrast (optional)
    
    Returns:
        Tuple (R, G, B) màu đã được điều chỉnh luminance
    """
    if background_color is None:
        # Nếu không có background, chỉ điều chỉnh luminance đơn giản
        h, s, v = rgb_to_hsv(rgb_color)
        # Điều chỉnh value để đạt target_luminance (xấp xỉ)
        # Tăng/giảm value dựa trên target
        current_lum = calculate_luminance(rgb_color)
        if current_lum < target_luminance:
            # Cần tăng luminance: tăng value
            new_v = min(1.0, v + (target_luminance - current_lum) * 2.0)
        else:
            # Cần giảm luminance: giảm value
            new_v = max(0.0, v - (current_lum - target_luminance) * 2.0)
        return hsv_to_rgb((h, s, new_v))
    
    current_lum = calculate_luminance(rgb_color)
    
    # Nếu đã đạt target và contrast đủ, không cần điều chỉnh
    if abs(current_lum - target_luminance) < 0.01:
        contrast = calculate_contrast_ratio(rgb_color, background_color)
        if contrast >= min_contrast:
            return rgb_color
    
    # Chuyển sang HSV để điều chỉnh value (luminance)
    h, s, v = rgb_to_hsv(rgb_color)
    
    # Binary search để tìm value tối ưu
    best_color = rgb_color
    best_contrast = calculate_contrast_ratio(rgb_color, background_color)
    best_lum_diff = abs(current_lum - target_luminance)
    
    # Tìm value tương ứng với target_luminance
    # Thử nhiều giá trị value và chọn giá trị gần target_luminance nhất với contrast >= min_contrast
    for test_v in np.linspace(0.0, 1.0, 50):  # Giảm số lần thử để tối ưu
        test_hsv = (h, s, test_v)
        test_rgb = hsv_to_rgb(test_hsv)
        test_lum = calculate_luminance(test_rgb)
        test_contrast = calculate_contrast_ratio(test_rgb, background_color)
        
        # Kiểm tra nếu gần target_luminance hơn và contrast đủ
        lum_diff = abs(test_lum - target_luminance)
        if test_contrast >= min_contrast:
            if lum_diff < best_lum_diff or (lum_diff == best_lum_diff and test_contrast > best_contrast):
                best_color = test_rgb
                best_contrast = test_contrast
                best_lum_diff = lum_diff
    
    # Nếu không tìm được màu đạt contrast, thử điều chỉnh nhẹ từ màu tốt nhất
    if best_contrast < min_contrast:
        # Điều chỉnh value để tăng contrast
        h, s, v = rgb_to_hsv(best_color)
        for adjustment in [0.05, 0.1, 0.15, 0.2, 0.3]:
            bg_lum = calculate_luminance(background_color)
            if bg_lum > 0.5:
                # Nền sáng: giảm value (làm tối hơn)
                new_v = max(0.0, v - adjustment)
            else:
                # Nền tối: tăng value (làm sáng hơn)
                new_v = min(1.0, v + adjustment)
            
            new_hsv = (h, s, new_v)
            new_rgb = hsv_to_rgb(new_hsv)
            new_contrast = calculate_contrast_ratio(new_rgb, background_color)
            
            if new_contrast > best_contrast:
                best_contrast = new_contrast
                best_color = new_rgb
                
                if best_contrast >= min_contrast:
                    return best_color
    
    return best_color

def choose_text_color(bg_color):
    """
    Chọn màu chữ dựa trên màu nền cụ thể để đảm bảo độ tương phản cao.
    Sử dụng WCAG contrast ratio >= 4.5 làm chuẩn.
    
    Args:
        bg_color: Tuple (R, G, B) màu nền
    
    Returns:
        Tuple (R, G, B) màu chữ có contrast tốt với nền
    """
    r, g, b = bg_color[0], bg_color[1], bg_color[2]
    
    # Định nghĩa các màu nền và màu chữ tương ứng
    # Sử dụng khoảng màu để nhận diện (không cần chính xác tuyệt đối)
    
    # Đỏ (#FF0000 hoặc gần đỏ): R cao, G và B thấp
    if r > 200 and g < 100 and b < 100:
        # Chọn xanh dương hoặc xanh lá
        text_color = (0, 0, 255)  # Xanh dương (#0000FF)
        if calculate_contrast_ratio(text_color, bg_color) < 4.5:
            text_color = (0, 255, 0)  # Xanh lá (#00FF00)
    
    # Xanh lá (#00FF00 hoặc gần xanh lá): G cao, R và B thấp
    elif g > 200 and r < 100 and b < 100:
        text_color = (255, 0, 0)  # Đỏ (#FF0000)
    
    # Xanh dương (#0000FF hoặc gần xanh dương): B cao, R và G thấp
    elif b > 200 and r < 100 and g < 100:
        text_color = (255, 255, 0)  # Vàng (#FFFF00)
    
    # Vàng (#FFFF00 hoặc gần vàng): R và G cao, B thấp
    elif r > 200 and g > 200 and b < 100:
        text_color = (0, 0, 255)  # Xanh dương (#0000FF)
    
    # Cyan (#00FFFF hoặc gần cyan): G và B cao, R thấp
    elif g > 200 and b > 200 and r < 100:
        text_color = (255, 0, 0)  # Đỏ (#FF0000)
    
    # Cam (#FFA500 hoặc gần cam): R cao, G trung bình, B thấp
    elif r > 200 and 100 < g < 200 and b < 100:
        text_color = (0, 0, 255)  # Xanh dương (#0000FF)
    
    # Tím (#800080 hoặc gần tím): R và B cao, G thấp
    elif r > 150 and b > 150 and g < 100:
        text_color = (255, 255, 0)  # Vàng (#FFFF00)
    
    # Đen hoặc gần đen: R, G, B đều thấp
    elif r < 50 and g < 50 and b < 50:
        text_color = (255, 255, 255)  # Trắng (#FFFFFF)
    
    # Trắng hoặc gần trắng: R, G, B đều cao
    elif r > 200 and g > 200 and b > 200:
        text_color = (0, 0, 0)  # Đen (#000000)
    
    # Kiểm tra contrast ratio và điều chỉnh nếu cần
    contrast = calculate_contrast_ratio(text_color, bg_color)
    
    if contrast < 4.5:
        # Nếu contrast không đạt, chỉ fallback giữa ĐEN và TRẮNG (không dùng thêm màu khác)
        bg_luminance = calculate_luminance(bg_color)
        
        black_contrast = calculate_contrast_ratio((0, 0, 0), bg_color)
        white_contrast = calculate_contrast_ratio((255, 255, 255), bg_color)
        
        if bg_luminance > 0.5:
            # Nền sáng: ưu tiên đen
            if black_contrast >= 4.5 or black_contrast >= white_contrast:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)
        else:
            # Nền tối: ưu tiên trắng
            if white_contrast >= 4.5 or white_contrast >= black_contrast:
                text_color = (255, 255, 255)
            else:
                text_color = (0, 0, 0)
    
    return text_color

CHAR_SPACING_THRESHOLD = 60
# Spacing nhẹ hơn: base nhỏ + tăng chậm + có cap để không bị “tách quá mạnh”
CHAR_SPACING_BASE_PX = 2
CHAR_SPACING_STEP_PX_PER = 20  # mỗi +20px font_size thì spacing tăng thêm 1px
CHAR_SPACING_MAX_PX = 6        # cap tối đa

def extra_char_spacing_px(font_size: int) -> int:
    """
    Trả về số pixel spacing thêm giữa các ký tự khi font lớn để tránh glyph dính nhau.
    """
    if font_size <= CHAR_SPACING_THRESHOLD:
        return 0
    extra = CHAR_SPACING_BASE_PX + max(0, (font_size - CHAR_SPACING_THRESHOLD) // CHAR_SPACING_STEP_PX_PER)
    return int(min(CHAR_SPACING_MAX_PX, extra))

def draw_text_with_conditional_spacing(
    draw,
    text: str,
    position,
    font,
    fill,
    *,
    is_multiline: bool = False,
    line_spacing: int = 0,
    stroke_width: int = 0,
    stroke_fill=None,
):
    """
    Vẽ text với spacing thủ công chỉ khi font lớn.
    Dùng cho cả ảnh và mask để đảm bảo alignment tuyệt đối.
    """
    x0, y0 = position
    try:
        fs = int(getattr(font, "size", 0) or 0)
    except Exception:
        fs = 0
    extra = extra_char_spacing_px(fs)

    # Không cần spacing thêm → dùng API native (nhanh hơn)
    if extra <= 0:
        if is_multiline:
            draw.multiline_text(
                (x0, y0),
                text,
                font=font,
                fill=fill,
                spacing=line_spacing,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        else:
            draw.text(
                (x0, y0),
                text,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        return

    lines = text.split("\n") if is_multiline else [text]
    y = y0
    for line in lines:
        x = x0
        for ch in line:
            draw.text(
                (x, y),
                ch,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            try:
                if hasattr(font, "getlength"):
                    ch_w = float(font.getlength(ch))
                else:
                    l, t, r, b = draw.textbbox((0, 0), ch, font=font)
                    ch_w = float(r - l)
            except Exception:
                ch_w = float(fs * 0.6) if fs > 0 else 10.0
            x += ch_w + extra

        if is_multiline:
            try:
                l, t, r, b = draw.textbbox((0, 0), line if line else "A", font=font)
                line_h = float(b - t)
            except Exception:
                line_h = float(fs) if fs > 0 else 20.0
            y += int(line_h + line_spacing)


def draw_text_with_outline(draw, text, position, font, bg_color, is_multiline=False, line_spacing=0):
    """
    Vẽ chữ lên canvas không viền (stroke_width=0).
    """
    text_color = choose_text_color(bg_color)
    outline_color = (0, 0, 0)
    outline_width = 0
    draw_text_with_conditional_spacing(
        draw, text, position, font, text_color,
        is_multiline=is_multiline, line_spacing=line_spacing,
        stroke_width=0, stroke_fill=None,
    )
    return text_color, outline_color, outline_width

def calculate_luminance(rgb):
    """
    Tính relative luminance theo WCAG 2.0 standard.
    Returns value between 0 (darkest) and 1 (lightest).
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # Linearize RGB values
    def linearize(val):
        if val <= 0.03928:
            return val / 12.92
        return ((val + 0.055) / 1.055) ** 2.4
    
    r_lin = linearize(r)
    g_lin = linearize(g)
    b_lin = linearize(b)
    
    # Calculate relative luminance
    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return luminance

def calculate_contrast_ratio(color1, color2):
    """
    Tính contrast ratio giữa 2 màu theo WCAG 2.0.
    Returns value >= 1.0 (1:1 = no contrast, 21:1 = maximum contrast).
    """
    lum1 = calculate_luminance(color1)
    lum2 = calculate_luminance(color2)
    
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    if darker == 0:
        return 21.0  # Maximum contrast
    
    return (lighter + 0.05) / (darker + 0.05)

def rgb_to_hsv(rgb):
    """Convert RGB (0-255) to HSV (0-1, 0-1, 0-1)."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb(hsv):
    """Convert HSV (0-1, 0-1, 0-1) to RGB (0-255)."""
    r, g, b = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    return (int(r * 255), int(g * 255), int(b * 255))

def get_contrast_color_hsv(background_color, min_contrast=4.5, rng=None):
    """
    Chọn màu chữ theo HSV/HSL với ràng buộc độ sáng để đảm bảo contrast với nền.
    Đảm bảo contrast >= 4.5, nếu không tự động chuyển sang đen/trắng.
    
    Ý tưởng:
    1. Tính bg_luma (độ sáng cảm nhận) của background
    2. Chọn hue ngẫu nhiên để đa dạng
    3. Ép lightness/value về phía đối nghịch với nền (nền sáng → chữ tối; nền tối → chữ sáng)
    4. Kiểm tra contrast ratio; nếu chưa đạt, điều chỉnh lightness thêm vài bước
    5. Nếu vẫn không đạt, fallback về đen/trắng
    
    Args:
        background_color: Tuple (R, G, B) của background
        min_contrast: Contrast ratio tối thiểu (default 4.5 cho WCAG AA)
        rng: Random generator (optional, để deterministic)
    
    Returns:
        Tuple (R, G, B) màu chữ có contrast tốt với background (>= min_contrast)
    """
    if rng is None:
        rng = random
    
    # Tính luminance của background
    bg_luminance = calculate_luminance(background_color)
    
    # Xác định hướng: nền sáng → chữ tối, nền tối → chữ sáng
    is_bg_light = bg_luminance > 0.5
    
    # Kiểm tra contrast với màu đen/trắng trước
    black_contrast = calculate_contrast_ratio((0, 0, 0), background_color)
    white_contrast = calculate_contrast_ratio((255, 255, 255), background_color)
    
    # Nếu đen hoặc trắng đã đạt min_contrast, có thể dùng trực tiếp
    if is_bg_light and black_contrast >= min_contrast:
        # Có thể thử màu khác nhưng đảm bảo contrast
        pass
    elif not is_bg_light and white_contrast >= min_contrast:
        # Có thể thử màu khác nhưng đảm bảo contrast
        pass
    
    # Chọn hue ngẫu nhiên để đa dạng màu sắc
    hue = rng.random()  # 0.0 đến 1.0
    
    # Saturation: giữ ở mức trung bình để màu rõ ràng nhưng không quá rực rỡ (0.4-0.8)
    # Giảm từ 0.5-1.0 xuống 0.4-0.8 để tránh màu quá chói
    saturation = 0.4 + rng.random() * 0.4
    
    # Thử nhiều giá trị lightness/value để đạt contrast tốt
    best_color = None
    best_contrast = 0.0
    
    # Điều chỉnh lightness dựa trên độ sáng của background
    if is_bg_light:
        # Background sáng: chữ cần tối (value thấp)
        # Bắt đầu từ value thấp và tăng dần nếu cần
        value_range = [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.05, 0.25)]  # Các khoảng value để thử
    else:
        # Background tối: chữ cần sáng (value cao)
        # Bắt đầu từ value cao và giảm dần nếu cần
        value_range = [(0.7, 0.9), (0.65, 0.85), (0.75, 0.95), (0.8, 1.0)]  # Các khoảng value để thử
    
    # Thử nhiều giá trị value trong các khoảng
    for value_min, value_max in value_range:
        for _ in range(10):  # 10 lần thử mỗi khoảng
            value = value_min + rng.random() * (value_max - value_min)
            
            # Convert HSV to RGB
            hsv = (hue, saturation, value)
            text_color = hsv_to_rgb(hsv)
            
            # Kiểm tra contrast ratio
            contrast = calculate_contrast_ratio(text_color, background_color)
            
            if contrast > best_contrast:
                best_contrast = contrast
                best_color = text_color
                
                # Nếu đạt min_contrast, điều chỉnh luminance và saturation để đảm bảo đủ sáng/tối và không quá rực rỡ
                if best_contrast >= min_contrast:
                    # Điều chỉnh luminance dựa trên is_bg_light
                    if is_bg_light:
                        # Nền sáng: chữ cần tối (target_luminance thấp, nhưng không quá tối)
                        target_lum = 0.1  # Đủ tối nhưng vẫn thấy được
                    else:
                        # Nền tối: chữ cần sáng (target_luminance cao)
                        target_lum = 0.9  # Đủ sáng nhưng không quá chói
                    
                    # Điều chỉnh luminance trước
                    adjusted_color = adjust_luminance(best_color, target_lum, min_contrast=min_contrast, background_color=background_color)
                    
                    # Sau đó giảm saturation nếu quá cao (để không quá rực rỡ)
                    adjusted_color = adjust_saturation(adjusted_color, max_saturation=0.7, min_contrast=min_contrast, background_color=background_color)
                    
                    # Kiểm tra lại contrast sau khi điều chỉnh
                    final_contrast = calculate_contrast_ratio(adjusted_color, background_color)
                    if final_contrast >= min_contrast:
                        return adjusted_color
                    return best_color
    
    # Nếu không đạt min_contrast, thử điều chỉnh thêm
    if best_color and best_contrast < min_contrast:
        # Điều chỉnh value để tăng contrast
        h, s, v = rgb_to_hsv(best_color)
        
        for adjustment in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if is_bg_light:
                # Giảm value (làm tối hơn)
                new_value = max(0.05, v - adjustment)
            else:
                # Tăng value (làm sáng hơn)
                new_value = min(0.95, v + adjustment)
            
            new_hsv = (h, s, new_value)
            new_color = hsv_to_rgb(new_hsv)
            new_contrast = calculate_contrast_ratio(new_color, background_color)
            
            if new_contrast > best_contrast:
                best_contrast = new_contrast
                best_color = new_color
                
                if best_contrast >= min_contrast:
                    return best_color
    
    # Nếu vẫn không đạt min_contrast, fallback về đen/trắng
    if is_bg_light:
        # Nền sáng: dùng đen
        if black_contrast >= min_contrast:
            # Đảm bảo chữ đen đủ tối (luminance thấp) nhưng vẫn có contrast tốt
            text_color = (0, 0, 0)
            # Điều chỉnh luminance để đảm bảo đủ tối (target_luminance = 0.0)
            text_color = adjust_luminance(text_color, 0.0, min_contrast=min_contrast, background_color=background_color)
            # Đen/trắng không cần điều chỉnh saturation (đã là màu trung tính)
            return text_color
        # Nếu đen không đạt, vẫn dùng đen (tốt nhất có thể)
        return (0, 0, 0)
    else:
        # Nền tối: dùng trắng
        if white_contrast >= min_contrast:
            # Đảm bảo chữ trắng đủ sáng (luminance cao)
            text_color = (255, 255, 255)
            # Điều chỉnh luminance để đảm bảo đủ sáng (target_luminance = 1.0)
            text_color = adjust_luminance(text_color, 1.0, min_contrast=min_contrast, background_color=background_color)
            # Đen/trắng không cần điều chỉnh saturation (đã là màu trung tính)
            return text_color
        # Nếu trắng không đạt, vẫn dùng trắng (tốt nhất có thể)
        return (255, 255, 255)

def get_simple_contrast_color(background_color, rng=None):
    """
    Tính màu chữ có contrast tốt với background sử dụng HSV/HSL.
    Sử dụng WCAG contrast ratio để đảm bảo màu chữ luôn thấy được.
    
    Args:
        background_color: Tuple (R, G, B) của background
        rng: Random generator (optional, để deterministic)
    
    Returns:
        Tuple (R, G, B) màu chữ có contrast tốt với background
    """
    return get_contrast_color_hsv(background_color, min_contrast=4.5, rng=rng)

def get_local_background_color(image_pil, x, y, w, h, padding=5):
    """
    Tính màu background trung bình tại vùng xung quanh vị trí text.
    Xử lý gradient và pattern bằng cách lấy mẫu nhiều điểm và tính median/mean.
    Điều này đảm bảo màu chữ contrast tốt với local background, không chỉ average toàn ảnh.
    
    Args:
        image_pil: PIL Image
        x, y: Vị trí text (top-left)
        w, h: Kích thước text
        padding: Padding xung quanh vùng text để tính background
    
    Returns:
        Tuple (R, G, B) màu background trung bình tại vùng này
    """
    try:
        # Convert to numpy
        img_np = np.array(image_pil.convert('RGB'))
        img_h, img_w = img_np.shape[:2]
        
        # Tính vùng để sample background (text area + padding)
        # Tăng padding để lấy mẫu tốt hơn cho gradient/pattern
        extended_padding = max(padding, int(min(w, h) * 0.1))
        x1 = max(0, int(x) - extended_padding)
        y1 = max(0, int(y) - extended_padding)
        x2 = min(img_w, int(x + w) + extended_padding)
        y2 = min(img_h, int(y + h) + extended_padding)
        
        # Lấy mẫu nhiều điểm để xử lý gradient/pattern tốt hơn
        # Thay vì chỉ lấy mean, lấy mẫu từ nhiều vị trí và dùng median để tránh outlier
        sample_region = img_np[y1:y2, x1:x2]
        
        if sample_region.size == 0:
            # Fallback: lấy average của toàn ảnh
            return tuple(img_np.mean(axis=(0, 1)).astype(int))
        
        # Lấy mẫu từ nhiều điểm trong vùng (grid sampling)
        h_sample, w_sample = sample_region.shape[:2]
        step_h = max(1, h_sample // 5)
        step_w = max(1, w_sample // 5)
        
        samples = []
        for sy in range(0, h_sample, step_h):
            for sx in range(0, w_sample, step_w):
                samples.append(sample_region[sy, sx])
        
        # Thêm samples từ border để tránh lấy mẫu từ vùng text
        # Top border
        if y1 >= 0 and y1 < img_h and x1 < x2:
            border_samples = img_np[y1, x1:x2]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Bottom border
        if y2 > 0 and y2 <= img_h and x1 < x2:
            border_samples = img_np[y2-1, x1:x2]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Left border
        if x1 >= 0 and x1 < img_w and y1 < y2:
            border_samples = img_np[y1:y2, x1]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Right border
        if x2 > 0 and x2 <= img_w and y1 < y2:
            border_samples = img_np[y1:y2, x2-1]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        
        if not samples:
            # Fallback: lấy mean của vùng
            return tuple(sample_region.mean(axis=(0, 1)).astype(int))
        
        # Thêm samples từ border để tránh lấy mẫu từ vùng text
        # Top border
        if y1 >= 0 and y1 < img_h and x1 < x2:
            border_samples = img_np[y1, x1:x2]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Bottom border
        if y2 > 0 and y2 <= img_h and x1 < x2:
            border_samples = img_np[y2-1, x1:x2]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Left border
        if x1 >= 0 and x1 < img_w and y1 < y2:
            border_samples = img_np[y1:y2, x1]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        # Right border
        if x2 > 0 and x2 <= img_w and y1 < y2:
            border_samples = img_np[y1:y2, x2-1]
            if border_samples.size > 0:
                samples.append(border_samples.mean(axis=0))
        
        if not samples:
            # Fallback: lấy mean của vùng
            return tuple(sample_region.mean(axis=(0, 1)).astype(int))
        
        # Dùng median để tránh ảnh hưởng của outlier (tốt hơn cho gradient/pattern)
        samples_array = np.array(samples)
        bg_color = tuple(np.median(samples_array, axis=0).astype(int))
        return bg_color
    except:
        # Fallback: tính average của toàn ảnh
        try:
            img_np = np.array(image_pil.convert('RGB'))
            return tuple(img_np.mean(axis=(0, 1)).astype(int).tolist())
        except:
            return (128, 128, 128)  # Gray fallback

def prepare_canvas_and_real_region(bg_image_pil, target_size=224):
    """
    Normalize background to target_size×target_size BEFORE drawing text.
    Returns both the canvas and the real background region coordinates.
    
    Rules:
    - If background is larger than target_size → resize with aspect ratio kept 
      so the longest side = target_size, then pad with white to target_size×target_size
    - If background is smaller → pad with white to target_size×target_size (no upscaling)
    
    Args:
        bg_image_pil: PIL Image (background)
        target_size: Target size (default 224)
    
    Returns:
        tuple: (canvas, real_region) where:
            canvas: PIL Image of size (target_size, target_size) ready for text rendering
            real_region: tuple (x1, y1, x2, y2) coordinates of real background area
                        (excluding white padding)
    """
    if bg_image_pil is None:
        # Return white canvas if None (no real region)
        canvas = Image.new('RGBA', (target_size, target_size), (255, 255, 255, 255))
        return canvas, (0, 0, target_size, target_size)  # Full canvas is "real" if no bg
    
    # Convert to RGBA if needed
    if bg_image_pil.mode != 'RGBA':
        bg_image_pil = bg_image_pil.convert('RGBA')
    
    w, h = bg_image_pil.size
    
    # Case 1: Background is larger than target_size
    if w > target_size or h > target_size:
        # Resize with aspect ratio kept so longest side = target_size
        if w > h:
            # Width is longer
            scale = target_size / w
            new_w = target_size
            new_h = int(h * scale)
        else:
            # Height is longer or equal
            scale = target_size / h
            new_h = target_size
            new_w = int(w * scale)
        
        # Resize background
        bg_resized = bg_image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create white canvas
        canvas = Image.new('RGBA', (target_size, target_size), (255, 255, 255, 255))
        
        # Center the resized background on canvas
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        canvas.paste(bg_resized, (x_offset, y_offset), bg_resized)
        
        # Real background region: (x1, y1, x2, y2)
        real_x1 = x_offset
        real_y1 = y_offset
        real_x2 = x_offset + new_w
        real_y2 = y_offset + new_h
        
        return canvas, (real_x1, real_y1, real_x2, real_y2)
    
    # Case 2: Background is smaller than or equal to target_size
    else:
        # No upscaling, just pad with white to target_size×target_size
        canvas = Image.new('RGBA', (target_size, target_size), (255, 255, 255, 255))
        
        # Center the background on canvas
        x_offset = (target_size - w) // 2
        y_offset = (target_size - h) // 2
        canvas.paste(bg_image_pil, (x_offset, y_offset), bg_image_pil)
        
        # Real background region: (x1, y1, x2, y2)
        real_x1 = x_offset
        real_y1 = y_offset
        real_x2 = x_offset + w
        real_y2 = y_offset + h
        
        return canvas, (real_x1, real_y1, real_x2, real_y2)

def place_text_inside_real_region(text, font_path, font_size, real_region, 
                                   margin=5, min_font_size=28, position_variation=True, max_stroke_for_fit=None):
    """
    Place text inside real background region with guaranteed no overflow.
    
    Args:
        text: Text string to place
        font_path: Path to font file
        font_size: Initial font size
        real_region: Tuple (real_x1, real_y1, real_x2, real_y2) of real background area
        margin: Margin from edges (default 5)
        min_font_size: Minimum font size to allow (default 28)
        position_variation: If True, randomly position; if False, center
    
    Returns:
        Tuple (x, y, font_size, font, w, h, l, t, is_multiline, line_spacing)
        Returns None if text cannot fit even at min_font_size
    """
    real_x1, real_y1, real_x2, real_y2 = real_region
    real_w = real_x2 - real_x1
    real_h = real_y2 - real_y1
    
    # Cache font to avoid reloading
    font_cache = {}
    
    # Try to fit text by shrinking font if needed
    current_font_size = font_size
    while current_font_size >= min_font_size:
        # Recalculate stroke and margin based on current font size
        # Stroke should scale with font size for better fit
        max_stroke = max(2, current_font_size // 20, 3)  # Account for stroke (scales with font size)
        effective_margin = margin + max_stroke
        
        try:
            # Cache fonts to avoid reloading
            if current_font_size not in font_cache:
                font_cache[current_font_size] = ImageFont.truetype(str(font_path), current_font_size)
            font = font_cache[current_font_size]
        except:
            return None
        
        # Calculate text bbox
        is_multiline = "\n" in text
        line_spacing = max(0, int(current_font_size * 0.2))
        # Conditional spacing: dùng đúng công thức global để đo bbox khớp với lúc vẽ
        extra = extra_char_spacing_px(current_font_size)

        def _measure_line_width(line: str) -> float:
            if extra <= 0:
                bbox = font.getbbox(line)
                return float(bbox[2] - bbox[0])
            total = 0.0
            for ch in line:
                try:
                    if hasattr(font, "getlength"):
                        ch_w = float(font.getlength(ch))
                    else:
                        cb = font.getbbox(ch)
                        ch_w = float(cb[2] - cb[0])
                except Exception:
                    ch_w = float(current_font_size * 0.6)
                total += ch_w + extra
            return total
        
        if is_multiline:
            # For multiline, calculate manually
            lines = text.split("\n")
            widths, heights = [], []
            for ln in lines:
                widths.append(_measure_line_width(ln))
                try:
                    bbox = font.getbbox(ln if ln else "A")
                    heights.append(bbox[3] - bbox[1])
                except Exception:
                    heights.append(current_font_size)
            w = max(widths) if widths else current_font_size
            h = sum(heights) + line_spacing * (len(heights) - 1) if heights else current_font_size
            # Với char-spacing, bbox offset không còn đáng tin → dùng l=t=0 để placement an toàn
            if extra > 0:
                l, t = 0, 0
            else:
                first_line_bbox = font.getbbox(lines[0])
                l = first_line_bbox[0]
                t = first_line_bbox[1]
        else:
            if extra > 0:
                w = _measure_line_width(text)
                try:
                    bbox = font.getbbox(text if text else "A")
                    h = bbox[3] - bbox[1]
                except Exception:
                    h = current_font_size
                l, t = 0, 0
            else:
                bbox = font.getbbox(text)
                l, t, r, b = bbox
                w = r - l
                h = b - t
        
        # Calculate allowed placement area (accounting for stroke)
        # Text bbox with stroke: (x + l - max_stroke, y + t - max_stroke) to (x + l + w + max_stroke, y + t + h + max_stroke)
        # Must be inside: (real_x1, real_y1) to (real_x2, real_y2)
        # So: x + l - max_stroke >= real_x1 + margin  => x >= real_x1 + margin - l + max_stroke
        # And: x + l + w + max_stroke <= real_x2 - margin  => x <= real_x2 - margin - w - l - max_stroke
        
        x_min = real_x1 + effective_margin - l + max_stroke
        y_min = real_y1 + effective_margin - t + max_stroke
        x_max = real_x2 - effective_margin - w - l - max_stroke
        y_max = real_y2 - effective_margin - h - t - max_stroke
        
        # Check if text fits
        if x_max >= x_min and y_max >= y_min:
            # Text fits! Choose position
            if position_variation:
                x = random.randint(int(x_min), int(x_max))
                y = random.randint(int(y_min), int(y_max))
            else:
                # Center
                x = (real_x1 + real_x2 - w) / 2 - l
                y = (real_y1 + real_y2 - h) / 2 - t
                # Clamp to valid range
                x = max(x_min, min(x, x_max))
                y = max(y_min, min(y, y_max))
            
            # Final verification: ensure text bbox with stroke is inside real_region
            text_left_with_stroke = x + l - max_stroke
            text_top_with_stroke = y + t - max_stroke
            text_right_with_stroke = x + l + w + max_stroke
            text_bottom_with_stroke = y + t + h + max_stroke
            
            # If still outside, clamp
            if text_left_with_stroke < real_x1:
                x = real_x1 - l + max_stroke
            if text_top_with_stroke < real_y1:
                y = real_y1 - t + max_stroke
            if text_right_with_stroke > real_x2:
                x = real_x2 - w - l - max_stroke
            if text_bottom_with_stroke > real_y2:
                y = real_y2 - h - t - max_stroke
            
            return (x, y, current_font_size, font, w, h, l, t, is_multiline, line_spacing)
        
        # Text doesn't fit, shrink font
        current_font_size -= 2
    
    # Text cannot fit even at min_font_size
    return None


def generate_white_text_on_black(text, font_path, font_size, image_size=(224, 224),
                                  position_variation=True, rng=None):
    """
    Tạo ảnh 100% chữ trắng trên nền đen. Không dùng background hay pattern.
    Returns:
        (image_np, mask_np) với image_np RGB uint8, mask_np float32 [0,1]; hoặc (None, None) nếu lỗi.
    """
    try:
        if rng is None:
            rng = random.Random(hash(text) % (2**31))
        real_region = (0, 0, image_size[0], image_size[1])
        placement_result = place_text_inside_real_region(
            text=text,
            font_path=font_path,
            font_size=font_size,
            real_region=real_region,
            margin=10,
            min_font_size=28,
            position_variation=position_variation,
            max_stroke_for_fit=0,
        )
        if placement_result is None:
            return None, None
        x, y, font_size, font, w, h, l, t, is_multiline, line_spacing = placement_result

        canvas = Image.new("RGB", image_size, (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw_text_with_conditional_spacing(
            draw, text, (x, y), font, (255, 255, 255),
            is_multiline=is_multiline, line_spacing=line_spacing,
            stroke_width=0, stroke_fill=None,
        )

        mask = Image.new("L", image_size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_text_with_conditional_spacing(
            draw_mask, text, (x, y), font, 255,
            is_multiline=is_multiline, line_spacing=line_spacing,
            stroke_width=0, stroke_fill=None,
        )
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        image_np = np.array(canvas, dtype=np.uint8)
        return image_np, mask_np
    except Exception:
        return None, None


def _generate_enhanced_image_from_memory(text, font_path, font_size,
                                        pattern_pil, background_pil,
                                        image_size=(224, 224),
                                        use_pattern_fill_prob=0.15,
                                        position_variation=True,
                                        real_region=None,
                                        rng=None,
                                        erode_gt_mask_px=None):
    try:
        # Tạo RNG nếu chưa có (fallback)
        if rng is None:
            text_seed = hash(text) % (2**31)
            rng = random.Random(text_seed)
        
        # Optimize: convert to RGB only once and cache
        if background_pil.mode != "RGB":
            bg_rgb = background_pil.convert("RGB")
        else:
            bg_rgb = background_pil
        bg_np_rgb = np.array(bg_rgb)
        # Tính average color của toàn background (fallback nếu không tính được local)
        avg_bg_color = tuple(bg_np_rgb.mean(axis=(0, 1)).astype(int))
        
        # Màu chữ sẽ được tính sau khi biết vị trí text (dùng local background color)
        # Tạm thời dùng average color với RNG để deterministic
        text_color = get_simple_contrast_color(avg_bg_color, rng=rng)
        # Outline luôn là màu đối lập với text_color để đảm bảo contrast tốt
        outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
        
        # Font will be loaded in place_text_inside_real_region, don't load here yet
        # outline_width and stroke_width will be recalculated after placement
        
        # Optimize: convert to RGBA only once
        if background_pil.mode != "RGBA":
            final_image_pil = background_pil.convert("RGBA")
        else:
            final_image_pil = background_pil
        draw = ImageDraw.Draw(final_image_pil)

        # Get real background region (where text can be placed)
        # If real_region is None, use full image (backward compatibility)
        if real_region is None:
            real_x1, real_y1, real_x2, real_y2 = 0, 0, image_size[0], image_size[1]
        else:
            real_x1, real_y1, real_x2, real_y2 = real_region
        
        # Ensure real_region is valid
        real_x1 = max(0, min(real_x1, image_size[0]))
        real_y1 = max(0, min(real_y1, image_size[1]))
        real_x2 = max(real_x1, min(real_x2, image_size[0]))
        real_y2 = max(real_y1, min(real_y2, image_size[1]))
        
        real_w = real_x2 - real_x1
        real_h = real_y2 - real_y1

        # --- BBOX & POSITION - Use place_text_inside_real_region for guaranteed placement ---
        try:
            # Use the new function to place text inside real region
            placement_result = place_text_inside_real_region(
                text=text,
                font_path=font_path,
                font_size=font_size,
                real_region=(real_x1, real_y1, real_x2, real_y2),
                margin=5,
                min_font_size=28,
                position_variation=position_variation,
                max_stroke_for_fit=0,
            )
            
            if placement_result is None:
                # Text cannot fit, return None
                return None
            
            x, y, font_size, font, w, h, l, t, is_multiline, line_spacing = placement_result
            
            # Tính local background color tại vị trí text để đảm bảo contrast tốt
            local_bg_color = get_local_background_color(
                final_image_pil, x + l, y + t, w, h, padding=5
            )
            # Tính lại màu chữ dựa trên local background color với RNG để deterministic
            text_color = get_simple_contrast_color(local_bg_color, rng=rng)
            # Outline luôn là màu đối lập với text_color để đảm bảo contrast tốt
            outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
            
            # Recalculate outline_width and stroke_width_for_pattern with final font_size
            # Đảm bảo độ dày outline hợp lý để chữ luôn rõ ràng
            outline_width = max(2, font_size // 20)
            stroke_width_for_pattern = max(3, font_size // 15)
            max_stroke = max(outline_width, stroke_width_for_pattern)
        except:
            # Fallback: simple placement
            w, h = len(text) * font_size * 0.6, font_size
            # Center within real_region, not full image
            x = (real_x1 + real_x2 - w) / 2
            y = (real_y1 + real_y2 - h) / 2
            
            # Tính local background color tại vị trí text
            try:
                local_bg_color = get_local_background_color(
                    final_image_pil, x, y, w, h, padding=5
                )
                # Dùng RNG để deterministic
                text_color = get_simple_contrast_color(local_bg_color, rng=rng)
                outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
            except:
                # Fallback về average color với RNG
                text_color = get_simple_contrast_color(avg_bg_color, rng=rng)
                outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
            
            if 'outline_width' not in locals():
                outline_width = 0
            if 'stroke_width_for_pattern' not in locals():
                stroke_width_for_pattern = 0
            is_multiline = "\n" in text
            line_spacing = max(0, int(font_size * 0.2))

        # Xác định có dùng pattern không (check một lần, dùng cho cả mask và ảnh render)
        use_pattern = random.random() < use_pattern_fill_prob
        
        # Đảm bảo stroke_width_for_pattern đã được tính (nếu chưa có)
        if 'stroke_width_for_pattern' not in locals():
            stroke_width_for_pattern = max(3, font_size // 15)
        
        # Xác định stroke width sẽ dùng cho mask (giống y hệt ảnh render)
        if use_pattern:
            stroke_width_mask = stroke_width_for_pattern  # Dùng stroke dày hơn cho pattern
        else:
            stroke_width_mask = 0  # Chữ vẽ stroke_width=0 → mask cũng 0 để GT khớp glyph, không nở (tránh mask dày hơn chữ)
        
        # Tạo GT mask bằng cách draw text trực tiếp với stroke giống y hệt ảnh render
        # CRITICAL: Mask phải có stroke giống ảnh render để seg loss học đúng
        mask = Image.new("L", image_size, 0)
        draw_mask = ImageDraw.Draw(mask)
        # Vẽ mask với stroke + conditional spacing giống y hệt ảnh render (fill + stroke)
        draw_text_with_conditional_spacing(
            draw_mask,
            text,
            (x, y),
            font,
            255,
            is_multiline=is_multiline,
            line_spacing=line_spacing,
            stroke_width=stroke_width_mask,
            stroke_fill=255,
        )
        
        if use_pattern:
            # Vẽ chữ không viền, sau đó fill pattern
            draw_text_with_conditional_spacing(
                draw, text, (x, y), font, text_color,
                is_multiline=is_multiline, line_spacing=line_spacing,
                stroke_width=0, stroke_fill=None,
            )
            pattern_with_alpha = pattern_pil.copy()
            pattern_with_alpha.putalpha(mask)
            final_image_pil = Image.alpha_composite(final_image_pil, pattern_with_alpha)
        else:
            # Vẽ chữ thường không viền
            draw_text_with_conditional_spacing(
                draw, text, (x, y), font, text_color,
                is_multiline=is_multiline, line_spacing=line_spacing,
                stroke_width=0, stroke_fill=None,
            )

        # Convert final image to numpy
        img_rgb = np.array(final_image_pil.convert("RGB"))

        # Convert mask từ PIL Image L (0-255) sang numpy float32 (0-1)
        # CRITICAL: Threshold về binary ngay từ đầu để đảm bảo mask sắc nét
        try:
            mask_np = np.array(mask, dtype=np.float32) / 255.0  # Normalize [0, 1]
            # Threshold về binary: giá trị > 0.5 = 1, còn lại = 0
            # Điều này đảm bảo mask luôn binary, không có giá trị trung gian
            mask_np = (mask_np > 0.5).astype(np.float32)  # Binary: 0 hoặc 1
            # Đảm bảo mask đúng kích thước IMAGE_SHAPE (224x224)
            target_h, target_w = IMAGE_SHAPE
            if mask_np.shape[0] != target_h or mask_np.shape[1] != target_w:
                mask_np = cv2.resize(
                    mask_np,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST  # Dùng NEAREST để giữ binary
                )
                # Sau resize, threshold lại để đảm bảo binary (resize có thể tạo giá trị trung gian)
                mask_np = (mask_np > 0.5).astype(np.float32)
            # Erode nhẹ để mask khớp độ dày với chữ (train/valid dùng riêng: ERODE_GT_MASK_PX_TRAIN / ERODE_GT_MASK_PX_VAL)
            erode_px = erode_gt_mask_px if erode_gt_mask_px is not None else ERODE_GT_MASK_PX_TRAIN
            if erode_px > 0 and mask_np.max() > 0:
                kernel = np.ones((3, 3), np.uint8)
                mask_uint8 = (mask_np * 255).astype(np.uint8)
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=erode_px)
                mask_np = (mask_uint8 > 0).astype(np.float32)
        except Exception:
            mask_np = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32)

        return img_rgb, mask_np
    except:
        return None, None

# --- 4. DATASET ---
class OnTheFlyFontDataset(Dataset):
    def __init__(self, font_paths, label_to_index,
                 pattern_paths, background_paths,
                 transform=None, samples_per_font=40,
                 image_size=(224, 224), is_validation=False,
                 words_list=None, channel_a_prob=0.6,
                 multiline_prob_2words=0.5, multiline_prob_letters=0.4,
                 skip_text_region_augment=False):
        self.font_paths = font_paths
        self.labels = [label_to_index.get(get_font_family_label(p), -1) for p in font_paths]
        valid_indices = [i for i, label in enumerate(self.labels) if label != -1]
        self.font_paths = [self.font_paths[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

        supported_font_paths = []
        supported_labels = []
        for fp, lb in zip(self.font_paths, self.labels):
            if font_supports_all_target_chars(fp):
                supported_font_paths.append(fp)
                supported_labels.append(lb)

        self.font_paths = supported_font_paths
        self.labels = supported_labels

        if not self.font_paths:
            raise RuntimeError("Không có font nào hỗ trợ đầy đủ bộ ký tự mục tiêu.")
        self.transform = transform
        self.image_size = image_size
        self.is_validation = is_validation
        self.skip_text_region_augment = skip_text_region_augment  # True khi train_deeplabv3 dùng augment riêng
        self.channel_a_prob = channel_a_prob  # Có thể cập nhật động mỗi epoch
        self.multiline_prob_2words = multiline_prob_2words  # Xác suất split 2 dòng khi có 2 từ
        self.multiline_prob_letters = multiline_prob_letters  # Xác suất split 2 dòng cho random letters
        # Không còn sử dụng words_list - tự động generate từ ngẫu nhiên
        self.words_list = []
        self.samples_list = []
        
        font_family_to_paths = {}
        for label_idx, font_path in zip(self.labels, self.font_paths):
            if label_idx not in font_family_to_paths:
                font_family_to_paths[label_idx] = []
            if os.path.exists(font_path):
                font_family_to_paths[label_idx].append(font_path)

        for label_idx, font_paths_list in font_family_to_paths.items():
            for _ in range(samples_per_font):
                selected_font_path = random.choice(font_paths_list)
                self.samples_list.append((label_idx, selected_font_path))

        random.shuffle(self.samples_list)
        print(f"Dataset {'Validation' if is_validation else 'Training'} tạo với {len(self.samples_list)} samples.")

        self.patterns_in_memory = []
        self.backgrounds_in_memory = []
        self.white_on_black_only = not pattern_paths and not background_paths

        if self.white_on_black_only:
            print("Chế độ 100% chữ trắng nền đen: không dùng pattern/background.")
        else:
            max_size = (MAX_GENERATED_WIDTH, MAX_GENERATED_HEIGHT)
            total_patterns = len(pattern_paths)
            total_backgrounds = len(background_paths)
            pattern_paths_sample = pattern_paths
            background_paths_sample = background_paths
            if len(pattern_paths_sample) > MAX_PATTERNS_IN_MEMORY:
                pattern_paths_sample = random.sample(pattern_paths_sample, MAX_PATTERNS_IN_MEMORY)
            if len(background_paths_sample) > MAX_BACKGROUNDS_IN_MEMORY:
                background_paths_sample = random.sample(background_paths_sample, MAX_BACKGROUNDS_IN_MEMORY)
            for path in tqdm(pattern_paths_sample, desc="Tải Patterns", leave=False):
                try:
                    img = Image.open(path).convert("RGBA").resize(max_size)
                    self.patterns_in_memory.append(img)
                except Exception:
                    pass
            for path in tqdm(background_paths_sample, desc="Tải Backgrounds", leave=False):
                try:
                    img = Image.open(path).convert("RGBA").resize(max_size)
                    self.backgrounds_in_memory.append(img)
                except Exception:
                    pass
            if not self.patterns_in_memory or not self.backgrounds_in_memory:
                raise RuntimeError("Không thể tải họa tiết/nền.")
            print(f"Đã tải {len(self.patterns_in_memory)}/{total_patterns} patterns và {len(self.backgrounds_in_memory)}/{total_backgrounds} backgrounds vào memory (giới hạn: {MAX_PATTERNS_IN_MEMORY}/{MAX_BACKGROUNDS_IN_MEMORY}).")

    def __len__(self):
        return len(self.samples_list)

    def _get_chars_for_sample(self, font_path, idx):
        """
        Deterministically generate characters for a sample based on (font_path, idx).
        This ensures consistent char selection across workers and epochs for better coverage.
        """
        # Use hash of (font_path, idx) as seed for deterministic generation
        # This ensures same (font_path, idx) always gets same chars, regardless of worker
        seed = hash((font_path, idx)) % (2**31)
        rng = random.Random(seed)
        
        # Rotate through TARGET_CHARS based on idx to ensure coverage
        # This ensures different samples use different subsets of chars
        char_rotation = idx % len(TARGET_CHARS)
        rotated_chars = TARGET_CHARS[char_rotation:] + TARGET_CHARS[:char_rotation]
        
        return rng, rotated_chars

    def __getitem__(self, idx):
        # Bọc toàn bộ logic trong try/except để tránh crash DataLoader worker
        try:
            try:
                label, font_path = self.samples_list[idx]
            except IndexError:
                return torch.randn(3, *self.image_size), -1

            # Get deterministic RNG and char rotation for this (font_path, idx)
            rng, rotated_chars = self._get_chars_for_sample(font_path, idx)
            
            # Lưu rng để dùng cho color generation sau này
            self._current_rng = rng

            # Tự động generate text: 85% từ ngẫu nhiên, 15% chữ cái ngẫu nhiên
            use_words = rng.random() < 0.85
            
            if use_words:
                # WORDS MODE (85%): 1-2 từ tự động generate
                num_words = rng.randint(1, 2)
                # Use rotated_chars to ensure coverage, but still random selection
                selected_words = []
                for _ in range(num_words):
                    word_length = rng.randint(3, 8)
                    # Select chars from rotated_chars to ensure coverage
                    word = ''.join(rng.choice(rotated_chars) for _ in range(word_length))
                    selected_words.append(word)
                
                # Join words with space, then decide if split into 2 lines
                text = ' '.join(selected_words)
                
                # Max 1 line break (so max 2 lines)
                # Randomly decide: single line or split into 2 lines
                # Dùng schedule probability (tăng dần về cuối để "lộ style")
                if len(selected_words) == 2 and rng.random() < self.multiline_prob_2words:
                    # Split 2 words into 2 lines
                    lines = selected_words
                else:
                    # Single line
                    lines = [text]
                
            else:
                # LETTERS MODE (15%): 4-5 random characters (not real words)
                num_chars = rng.randint(4, 5)
                
                # Select characters from rotated_chars to ensure coverage
                # Prioritize chars from the rotated set to ensure diversity
                selected_chars = [rng.choice(rotated_chars) for _ in range(num_chars)]
                
                # Max 1 line break (so max 2 lines)
                # Randomly decide: single line or split into 2 lines
                # Dùng schedule probability (tăng dần về cuối để "lộ style")
                if rng.random() < self.multiline_prob_letters:
                    # Split into 2 lines evenly
                    mid_point = len(selected_chars) // 2
                    lines = [
                        ''.join(selected_chars[:mid_point]),
                        ''.join(selected_chars[mid_point:])
                    ]
                else:
                    # Single line
                    lines = [''.join(selected_chars)]
            
            # Join lines with newline character (max 1 newline = max 2 lines)
            word = '\n'.join(lines)

            # Font size: directly use target size (no scaling needed since canvas is already 224×224)
            # Use deterministic RNG for consistent results
            font_size = rng.randint(28, 150)
            
            if self.white_on_black_only:
                image_np, mask_np = generate_white_text_on_black(
                    text=word,
                    font_path=font_path,
                    font_size=font_size,
                    image_size=IMAGE_SHAPE,
                    position_variation=True,
                    rng=rng,
                )
            else:
                pattern_pil = rng.choice(self.patterns_in_memory)
                background_pil = rng.choice(self.backgrounds_in_memory)
                current_w = rng.randint(MIN_GENERATED_WIDTH, MAX_GENERATED_WIDTH)
                current_h = rng.randint(MIN_GENERATED_HEIGHT, MAX_GENERATED_HEIGHT)
                pattern_pil = crop_region_pil(pattern_pil.copy(), current_w, current_h)
                background_pil = crop_region_pil(background_pil.copy(), current_w, current_h)
                background_pil, real_region = prepare_canvas_and_real_region(
                    background_pil, target_size=IMAGE_SHAPE[0]
                )
                pattern_pil, _ = prepare_canvas_and_real_region(
                    pattern_pil, target_size=IMAGE_SHAPE[0]
                )
                erode_px = ERODE_GT_MASK_PX_VAL if self.is_validation else ERODE_GT_MASK_PX_TRAIN
                image_np, mask_np = _generate_enhanced_image_from_memory(
                    text=word, font_path=font_path, font_size=font_size,
                    pattern_pil=pattern_pil, background_pil=background_pil,
                    image_size=IMAGE_SHAPE,
                    real_region=real_region,
                    rng=rng,
                    erode_gt_mask_px=erode_px,
                )

            if image_np is None or mask_np is None:
                dummy_img = torch.randn(3, *self.image_size)
                dummy_mask = torch.zeros(1, *self.image_size, dtype=torch.float32)
                return dummy_img, -1, dummy_mask

            # Augment chỉ khi train và không bị tắt (train_deeplabv3 dùng augment riêng → skip_text_region_augment=True)
            if not self.is_validation and not self.skip_text_region_augment:
                image_np = apply_text_region_augmentations(image_np, mask_np, rng)

            # Image is already 224×224, so transform should NOT resize
            # Only apply augmentation (rotation, brightness, etc.)
            # IMPORTANT: Augment mask đồng bộ với image để seg loss học đúng
            # CRITICAL: Mask KHÔNG bị ảnh hưởng bởi A.Normalize (ImageNet normalization)
            #           A.Normalize chỉ áp dụng cho image, mask được xử lý riêng
            if self.transform:
                # Convert mask từ [0, 1] float32 sang [0, 255] uint8 để augment (Albumentations yêu cầu)
                # Hoặc giữ float [0, 1] - Albumentations hỗ trợ cả hai, nhưng uint8 chắc chắn hơn
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)
                
                # Augment đồng bộ: các transform geometric (Rotate, Affine) sẽ áp dụng cho cả image và mask
                # CRITICAL: Các transform như A.Normalize, A.RandomBrightnessContrast, A.GaussNoise, 
                #           A.ISONoise, A.ImageCompression CHỈ áp dụng cho image, KHÔNG áp dụng cho mask
                augmented = self.transform(image=image_np, mask=mask_np_uint8)
                image_tensor = augmented['image'].float()  # Image đã được normalize theo ImageNet
                
                # Lấy mask đã được augment và convert lại về [0, 1] float32
                # IMPORTANT: ToTensorV2() convert mask thành torch.Tensor từ uint8 [0,255] → float [0,1]
                #           Mask KHÔNG bị normalize theo ImageNet mean/std (chỉ image mới bị normalize)
                mask_augmented = augmented['mask']
                
                if isinstance(mask_augmented, torch.Tensor):
                    # Mask đã là torch.Tensor (sau ToTensorV2)
                    # ToTensorV2 convert uint8 [0, 255] thành float [0, 1], shape: [H, W]
                    # CRITICAL: Mask KHÔNG bị normalize theo ImageNet mean/std (chỉ image mới bị normalize)
                    #           ToTensorV2 chỉ convert [0,255] → [0,1], không áp dụng mean/std
                    mask_tensor = mask_augmented.float()  # Đảm bảo float32
                    if mask_tensor.max() > 1.0:
                        # Nếu vẫn còn [0, 255], normalize về [0, 1] (chỉ scale, không dùng ImageNet mean/std)
                        mask_tensor = mask_tensor / 255.0
                    # CRITICAL: Threshold về binary để giữ mask sắc nét (không bị blur)
                    # Sau khi augment với INTER_NEAREST, vẫn có thể có giá trị trung gian do rounding
                    # Đảm bảo mask luôn binary [0, 1], không có giá trị trung gian
                    mask_tensor = (mask_tensor > 0.5).float()  # Binary: 0 hoặc 1
                    mask_tensor = mask_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
                elif isinstance(mask_augmented, np.ndarray):
                    # Mask vẫn là numpy array (không có ToTensorV2)
                    if mask_augmented.dtype == np.uint8:
                        mask_np = mask_augmented.astype(np.float32) / 255.0
                    else:
                        mask_np = mask_augmented.astype(np.float32)
                    # CRITICAL: Threshold về binary trước khi convert sang tensor
                    mask_np = (mask_np > 0.5).astype(np.float32)  # Binary: 0 hoặc 1
                    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
                else:
                    # Fallback: nếu có type khác, thử convert
                    mask_np = np.array(mask_augmented, dtype=np.float32)
                    if mask_np.max() > 1.0:
                        mask_np = mask_np / 255.0
                    # CRITICAL: Threshold về binary
                    mask_np = (mask_np > 0.5).astype(np.float32)  # Binary: 0 hoặc 1
                    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
                
                # CRITICAL: Đảm bảo mask luôn binary [0, 1] sau mọi transform
                # Mask KHÔNG bị ảnh hưởng bởi A.Normalize (ImageNet normalization)
                # Mask chỉ được convert từ uint8 [0,255] → float [0,1] bởi ToTensorV2, không normalize theo mean/std
                assert mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0, \
                    f"Mask values out of range [0, 1]: min={mask_tensor.min()}, max={mask_tensor.max()}"
                # Đảm bảo mask binary (chỉ 0 hoặc 1, không có giá trị trung gian)
                mask_tensor = (mask_tensor > 0.5).float()  # Final threshold để đảm bảo binary
            else:
                # If no transform, just normalize to [0, 1]
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
                # Mask: [H, W] -> [1, H, W]
                # CRITICAL: Threshold về binary để đảm bảo mask sắc nét
                # Mask KHÔNG bị normalize theo ImageNet mean/std (chỉ image mới bị normalize)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
                mask_tensor = (mask_tensor > 0.5).float()  # Binary: 0 hoặc 1
                
                # CRITICAL: Đảm bảo mask luôn binary [0, 1]
                assert mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0, \
                    f"Mask values out of range [0, 1]: min={mask_tensor.min()}, max={mask_tensor.max()}"

            return image_tensor, label, mask_tensor
        except Exception:
            # Nếu bất kỳ lỗi nào xảy ra (font hỏng, pattern lỗi, augment lỗi, ...),
            # trả về ảnh dummy và label -1 để collate_fn lọc bỏ, tránh crash.
            dummy_img = torch.randn(3, *self.image_size)
            dummy_mask = torch.zeros(1, *self.image_size, dtype=torch.float32)
            return dummy_img, -1, dummy_mask

# --- 5. AUGMENTATION ---
# NOTE: Images are already 224×224 from prepare_canvas(), so NO resize needed
# Only apply augmentation transforms

def ensure_size(image, **kwargs):
    """Ensure image is exactly 224×224 (should already be, but double-check)."""
    target_h, target_w = IMAGE_SHAPE
    img = image.copy()
    h, w = img.shape[:2]
    
    # If already correct size, return as-is
    if h == target_h and w == target_w:
        return img
    
    # Otherwise, resize to target (shouldn't happen, but safety check)
    if h != target_h or w != target_w:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        temp = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        temp[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return temp
    
    return img

def ensure_size_mask(mask, **kwargs):
    """
    Ensure mask is exactly 224×224 with INTER_NEAREST to preserve binary values.
    CRITICAL: Always use INTER_NEAREST for mask resize to prevent blur/interpolation artifacts.
    """
    target_h, target_w = IMAGE_SHAPE
    
    # Handle different mask formats (Albumentations passes numpy array)
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        # Float mask [0, 1] - convert to uint8 for resize, then back
        mask_uint8 = (mask * 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        mask_uint8 = mask
    else:
        # Convert to uint8
        mask_uint8 = mask.astype(np.uint8)
    
    h, w = mask_uint8.shape[:2]
    
    # If already correct size, return as-is (but ensure binary)
    if h == target_h and w == target_w:
        mask_float = mask_uint8.astype(np.float32) / 255.0
        # CRITICAL: Threshold to binary to remove any intermediate values
        return (mask_float > 0.5).astype(np.float32)  # Binary: 0 or 1
    
    # Otherwise, resize to target with INTER_NEAREST (CRITICAL: no bilinear!)
    if h != target_h or w != target_w:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        # CRITICAL: Use INTER_NEAREST to preserve binary values (no blur)
        resized = cv2.resize(mask_uint8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        temp = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        temp[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        # Convert to float [0, 1] and threshold to binary
        mask_float = temp.astype(np.float32) / 255.0
        # CRITICAL: Threshold after resize to ensure binary (even with INTER_NEAREST)
        return (mask_float > 0.5).astype(np.float32)  # Binary: 0 or 1
    
    # Convert to float [0, 1] and threshold
    mask_float = mask_uint8.astype(np.float32) / 255.0
    return (mask_float > 0.5).astype(np.float32)  # Binary: 0 or 1

class EnsureSizeTransform(DualTransform):
    """
    Safety check: ensure image and mask are 224×224.
    Image uses INTER_AREA, mask uses INTER_NEAREST to preserve binary values.
    """
    def __init__(self, p: float = 1.0):
        super().__init__(p=p, always_apply=True)

    def apply(self, image, **params):
        return ensure_size(image)

    def apply_to_mask(self, mask, **params):
        return ensure_size_mask(mask)

# train_transform: chỉ Rotate toàn ảnh (JPEG/blur/downscale đã áp trên patch chữ)
_initial_aug_params = {'rotate_limit': 15, 'affine_scale': (0.94, 1.02)}
train_transform = build_train_transform(_initial_aug_params)

# Validation: không augment, chỉ Normalize + ToTensorV2
val_transform_no_aug = A.Compose([
    EnsureSizeTransform(p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # CHỈ cho image
    ToTensorV2()  # ToTensorV2 convert mask từ uint8 [0,255] → float [0,1], KHÔNG normalize theo ImageNet
])

def collate_fn(batch):
    """
    Collate cho batch có thể chứa (image, label, mask).
    Lọc các sample có label = -1 (dummy) và stack image/mask.
    """
    # Filter invalid samples
    batch = [b for b in batch if len(b) >= 2 and b[1] != -1]
    if not batch:
        empty = torch.tensor([])
        return empty, empty, empty

    # Mặc định dataset giờ trả (image, label, mask)
    if len(batch[0]) == 3:
        images, labels, masks = zip(*batch)
        images = torch.stack(images).float()
        labels = torch.tensor(labels, dtype=torch.long)
        masks = torch.stack(masks).float()
        return images, labels, masks
    else:
        # Backward compatibility: chỉ (image, label)
        images, labels = torch.utils.data.dataloader.default_collate(batch)
        images = images.float()
        masks = torch.zeros(len(labels), 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], dtype=torch.float32)
        return images, labels, masks

# --- 6. TẢI DATA ---

def _list_images_with_exts(dir_path, allowed_exts):
    results = []
    for root_dir, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.lower().endswith(allowed_exts):
                results.append(os.path.join(root_dir, filename))
    return results

def get_font_family_label(font_path, font_base_dir=None):
    """
    Trả về label là tên folder chứa font file.
    Tất cả các file font trong cùng một folder sẽ có cùng một label (class).
    Model sẽ học từ TẤT CẢ các file font trong folder (Regular, Bold, Italic, v.v.).
    """
    parent_dir = os.path.abspath(os.path.dirname(font_path))
    parent_dir_name = os.path.basename(parent_dir)
    
    # Nếu font nằm trực tiếp trong base_dir (không có subfolder), dùng tên file
    if font_base_dir:
        base_dir_path = os.path.abspath(font_base_dir)
        base_dir_name = os.path.basename(base_dir_path)
        if parent_dir_name == base_dir_name:
            # File ở root của font_dir, dùng tên file (không có extension)
            return os.path.splitext(os.path.basename(font_path))[0]
    
    # File nằm trong subfolder, dùng tên folder làm label
    return parent_dir_name

# GradScaler cho Mixed Precision Training (AMP)
scaler = GradScaler()
class EarlyStopping:
    def __init__(self, patience=100, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def top_k_accuracy(outputs, labels, k=5):
    with torch.no_grad():
        _, top_k = outputs.topk(k, dim=1)
        correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
        return correct.sum().float() / labels.size(0)

# --- HÀM HỖ TRỢ CHO VISUALIZATION ---
def get_font_path(font_name, font_dir=FONT_VAL_DIR):
    """
    Tìm đường dẫn font từ tên font (tên folder).
    Với cấu trúc label_20k_cleared/Arial/Arial-Bold.ttf, tìm file font trong folder font_name.
    Ưu tiên file Regular/Normal hoặc file có tên gần giống với tên folder.
    """
    # Tìm folder chứa font
    font_folder = os.path.join(font_dir, font_name)
    if not os.path.isdir(font_folder):
        # Fallback: thử tìm như file trực tiếp (cho trường hợp font ở root)
        for ext in ['.ttf', '.TTF', '.otf', '.OTF']:
            p = os.path.join(font_dir, font_name + ext)
            if os.path.exists(p):
                return p
        return None
    
    # Tìm tất cả file font trong folder
    font_files = []
    for file in os.listdir(font_folder):
        if file.lower().endswith(('.ttf', '.otf')):
            font_files.append(os.path.join(font_folder, file))
    
    if not font_files:
        return None
    
    # Ưu tiên file Regular, Normal, hoặc file có tên giống với folder
    font_name_lower = font_name.lower()
    priority_keywords = ['regular', 'normal', font_name_lower]
    
    # Tìm file có keyword ưu tiên
    for keyword in priority_keywords:
        for font_file in font_files:
            file_name_lower = os.path.basename(font_file).lower()
            if keyword in file_name_lower:
                return font_file
    
    # Nếu không tìm thấy, trả về file đầu tiên
    return font_files[0]

def create_prediction_visualization(image_tensor, top1_font_name, top5_indices, top5_probs, 
                                   gt_label, pred_label, index_to_label, font_dir=FONT_VAL_DIR,
                                   image_width=224, alphabet_width=224):
    """
    Tạo ảnh visualization: ảnh dự đoán bên trái + bảng 52 chữ cái (Top-1) bên phải (liền mạch)
    Vẽ Top-5 confidences trực tiếp lên ảnh
    
    Args:
        image_tensor: Tensor ảnh [C, H, W] đã denormalized
        top1_font_name: Tên font Top-1
        top5_indices: List 5 indices của Top-5 predictions
        top5_probs: List 5 probabilities của Top-5 predictions
        gt_label: Ground truth label (string)
        pred_label: Predicted label (string)
        index_to_label: Mapping từ index sang tên font
        font_dir: Thư mục chứa font
        image_width: Kích thước ảnh (chiều rộng)
        alphabet_width: Kích thước bảng chữ cái (chiều rộng)
    
    Returns:
        PIL Image đã được ghép (ảnh trái + bảng chữ cái phải, liền mạch)
    """
    try:
        # Convert tensor sang numpy và PIL
        img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_h, img_w = img_pil.size[1], img_pil.size[0]
        
        # Tạo bảng 52 chữ cái
        alpha_img = create_alphabet_grid(top1_font_name, width=alphabet_width, height=alphabet_width, font_dir=font_dir)
        alpha_pil = Image.fromarray(alpha_img)
        
        # Resize bảng chữ cái để match với chiều cao ảnh (liền mạch)
        alpha_pil = alpha_pil.resize((alphabet_width, img_h), Image.Resampling.LANCZOS)
        
        # Tạo ảnh mới: chỉ ảnh + bảng chữ cái (liền mạch, không có khoảng trống)
        total_width = img_w + alphabet_width
        total_height = img_h
        
        combined_img = Image.new('RGB', (total_width, total_height), 'white')
        
        # Paste ảnh gốc vào bên trái
        combined_img.paste(img_pil, (0, 0))
        
        # Paste bảng chữ cái ngay bên phải ảnh (liền mạch)
        combined_img.paste(alpha_pil, (img_w, 0))
        
        # Vẽ thông tin GT, Pred và Top-5 confidences trực tiếp lên ảnh (góc trên bên trái)
        draw = ImageDraw.Draw(combined_img)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Vẽ nền semi-transparent cho text
        overlay = Image.new('RGBA', combined_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        text_box_height = 160  # Đủ cho GT, Pred và 5 dòng Top-5
        overlay_draw.rectangle([(5, 5), (img_w - 5, text_box_height)], fill=(0, 0, 0, 180))
        combined_img = Image.alpha_composite(combined_img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(combined_img)
        
        x_start = 10
        y_start = 10
        line_height = 18
        
        # Vẽ GT và Pred
        is_correct = "✓" if gt_label == pred_label else "✗"
        color = (0, 255, 0) if gt_label == pred_label else (255, 0, 0)
        
        draw.text((x_start, y_start), f"GT: {gt_label[:18]}", fill=(255, 255, 255), font=font_large)
        y_start += line_height
        draw.text((x_start, y_start), f"Pred: {pred_label[:18]} {is_correct}", fill=color, font=font_large)
        y_start += line_height + 5
        
        # Vẽ Top-5 với confidences
        draw.text((x_start, y_start), "Top-5:", fill=(255, 255, 255), font=font_medium)
        y_start += line_height
        
        for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
            font_name = index_to_label.get(idx, f"Class_{idx}")
            font_name_short = font_name[:15] + "..." if len(font_name) > 15 else font_name
            prob_percent = prob * 100
            
            # Màu khác nhau cho mỗi rank
            if rank == 1:
                text_color = (255, 200, 100)  # Vàng nhạt cho Top-1
            elif rank == 2:
                text_color = (200, 200, 255)  # Xanh nhạt cho Top-2
            elif rank == 3:
                text_color = (200, 255, 200)  # Xanh lá nhạt cho Top-3
            else:
                text_color = (255, 255, 255)  # Trắng cho Top-4, 5
            
            text = f"{rank}. {font_name_short}: {prob_percent:.1f}%"
            draw.text((x_start, y_start), text, fill=text_color, font=font_small)
            y_start += line_height - 2
        
        return np.array(combined_img)
    except Exception as e:
        print(f"Lỗi tạo visualization: {e}")
        import traceback
        traceback.print_exc()
        return np.ones((224, 448, 3), dtype=np.uint8) * 255

def create_alphabet_grid(font_name, width=1000, height=1000, font_dir=FONT_VAL_DIR):
    """
    Tạo bảng 52 chữ cái: A-Z + a-z với khoảng cách rõ ràng, không bị dính vào nhau
    """
    try:
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 52 chữ cái: A-Z + a-z
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        cols = 13  # 13 chữ mỗi hàng
        rows = 4   # 4 hàng → 52 chữ
        
        # Tính toán cell size với padding để tránh chữ dính vào nhau
        padding = 10  # Padding giữa các chữ
        title_height = 50  # Chiều cao cho tiêu đề
        available_height = height - title_height - padding * 2
        available_width = width - padding * 2
        
        cell_w = available_width // cols
        cell_h = available_height // rows
        
        # Font size - nhỏ hơn một chút để đảm bảo có khoảng cách
        base_font_size = int(min(cell_w * 0.6, cell_h * 0.6))  # Giảm từ 1.25 và 1.0 xuống 0.6
        base_font_size = max(28, min(base_font_size, 80))  # Giới hạn từ 28 đến 80
        
        font_path = get_font_path(font_name, font_dir)
        default_font = ImageFont.load_default()
        
        # Tiêu đề
        title = f"Top-1: {font_name}"
        title_font_size = min(24, width // 30)
        try:
            if font_path:
                title_font = ImageFont.truetype(str(font_path), title_font_size)
            else:
                title_font = default_font
        except:
            title_font = default_font
            
        try:
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_w = title_bbox[2] - title_bbox[0]
        except:
            title_w = 400
        draw.text(((width - title_w) // 2, 10), title, fill='#1a1a1a', font=title_font)
        
        start_y = title_height + padding
        
        for idx, char in enumerate(chars):
            col = idx % cols
            row = idx // cols
            
            # Tính vị trí cell với padding
            cell_x = padding + col * cell_w
            cell_y = start_y + row * cell_h
            cx = cell_x + cell_w // 2
            cy = cell_y + cell_h // 2
            
            # Thử với font size ban đầu
            try:
                if font_path:
                    test_font = ImageFont.truetype(str(font_path), base_font_size)
                else:
                    test_font = default_font
                bbox = draw.textbbox((0, 0), char, font=test_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                tw, th = 30, 30
                test_font = default_font
            
            # Tự động scale để fit vào cell với margin
            margin = 8  # Margin trong mỗi cell
            max_w = cell_w - margin * 2
            max_h = cell_h - margin * 2
            
            scale_w = max_w / max(tw, 1)
            scale_h = max_h / max(th, 1)
            scale = min(scale_w, scale_h, 1.0)  # Không scale lên, chỉ scale xuống
            
            if scale < 1:
                new_size = max(28, int(base_font_size * scale))
                try:
                    if font_path:
                        test_font = ImageFont.truetype(str(font_path), new_size)
                    else:
                        test_font = default_font
                    bbox = draw.textbbox((0, 0), char, font=test_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except:
                    test_font = default_font
                    bbox = draw.textbbox((0, 0), char, font=test_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Căn giữa trong cell
            x = cx - tw // 2
            y = cy - th // 2
            
            # Đảm bảo không vượt quá biên
            x = max(padding + margin, min(x, width - padding - margin - tw))
            y = max(start_y + margin, min(y, height - padding - margin - th))
            
            # Vẽ chữ với màu đen rõ ràng
            draw.text((x, y), char, fill='#000000', font=test_font)
        
        return np.array(img)
    
    except Exception as e:
        print(f"Lỗi tạo bảng chữ: {e}")
        import traceback
        traceback.print_exc()
        return np.ones((height, width, 3), dtype=np.uint8) * 255

def draw_predictions_on_images(images_tensor, labels_tensor, preds_tensor, top5_preds_tensor, 
                                index_to_label=None, max_label_length=20):
    """
    Vẽ predictions trực tiếp lên ảnh (GT và Predicted labels)
    
    Args:
        images_tensor: Tensor ảnh (N, C, H, W) đã được denormalize [0, 1]
        labels_tensor: Tensor labels (N,)
        preds_tensor: Tensor predictions (N,)
        top5_preds_tensor: Tensor top-5 predictions (N, 5)
        index_to_label: Mapping từ index sang tên font
        max_label_length: Độ dài tối đa của label để hiển thị
    
    Returns:
        List các ảnh PIL đã được annotate
    """
    annotated_images = []
    
    for i in range(len(images_tensor)):
        # Convert tensor to numpy
        img_np = images_tensor[i].permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Get labels
        gt_idx = labels_tensor[i].item()
        pred_idx = preds_tensor[i].item()
        top5_indices = top5_preds_tensor[i].numpy()
        
        if index_to_label:
            gt_label = index_to_label.get(gt_idx, f"Class_{gt_idx}")
            pred_label = index_to_label.get(pred_idx, f"Class_{pred_idx}")
            top5_labels = [index_to_label.get(idx, f"Class_{idx}") for idx in top5_indices]
        else:
            gt_label = f"Class_{gt_idx}"
            pred_label = f"Class_{pred_idx}"
            top5_labels = [f"Class_{idx}" for idx in top5_indices]
        
        # Rút ngắn tên font nếu quá dài
        gt_label_short = gt_label[:max_label_length] + "..." if len(gt_label) > max_label_length else gt_label
        pred_label_short = pred_label[:max_label_length] + "..." if len(pred_label) > max_label_length else pred_label
        
        # Kiểm tra đúng/sai
        is_correct = (gt_idx == pred_idx)
        status_color = (0, 255, 0) if is_correct else (255, 0, 0)  # Xanh nếu đúng, đỏ nếu sai
        
        # Vẽ annotations
        draw = ImageDraw.Draw(img_pil)
        try:
            # Dùng font mặc định
            font_small = ImageFont.load_default()
            # Thử dùng font lớn hơn nếu có
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except:
                try:
                    font_large = ImageFont.load_default()
                except:
                    font_large = ImageFont.load_default()
        except:
            font_small = ImageFont.load_default()
            font_large = ImageFont.load_default()
        
        h, w = img_pil.size[1], img_pil.size[0]
        line_height = 12
        y_offset = 5
        
        # Tính số dòng cần thiết: GT, Pred, và Top5 (5 dòng)
        num_lines = 2 + len(top5_labels)  # GT + Pred + Top5
        text_bg_height = line_height * num_lines + 6
        
        # Vẽ nền cho text (semi-transparent)
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, 0), (w, text_bg_height)], fill=(0, 0, 0, 180))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Vẽ GT label (màu trắng)
        gt_text = f"GT: {gt_label_short}"
        draw.text((5, y_offset), gt_text, fill=(255, 255, 255), font=font_large)
        
        # Vẽ Predicted label với màu theo đúng/sai
        pred_text = f"Pred: {pred_label_short}"
        draw.text((5, y_offset + line_height), pred_text, fill=status_color, font=font_large)
        
        # Vẽ Top-5 predictions
        current_y = y_offset + line_height * 2
        for rank, top_label in enumerate(top5_labels, 1):
            # Rút ngắn tên font nếu quá dài
            top_label_short = top_label[:max_label_length] + "..." if len(top_label) > max_label_length else top_label
            # Màu khác nhau cho mỗi rank
            if rank == 1:
                rank_color = (255, 255, 0)  # Vàng cho top-1
            elif rank == 2:
                rank_color = (255, 200, 0)  # Cam vàng cho top-2
            elif rank == 3:
                rank_color = (200, 200, 255)  # Xanh nhạt cho top-3
            elif rank == 4:
                rank_color = (200, 255, 200)  # Xanh lá nhạt cho top-4
            else:
                rank_color = (255, 200, 200)  # Hồng nhạt cho top-5
            
            top_text = f"Top{rank}: {top_label_short}"
            draw.text((5, current_y), top_text, fill=rank_color, font=font_small)
            current_y += line_height
        
        annotated_images.append(img_pil)
    
    return annotated_images

def visualize_random_samples(model, loader, device, writer, tag_prefix, epoch, num_samples=20, index_to_label=None):
    """
    Visualize ngẫu nhiên num_samples ảnh từ loader lên TensorBoard với predictions được vẽ trực tiếp lên ảnh
    """
    model.eval()
    all_images = []
    all_labels = []
    all_preds = []
    all_top5_preds = []
    
    with torch.no_grad():
        for batch in loader:
            # Loader có thể trả (images, labels, masks)
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            if images.nelement() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Model có thể trả tuple (main_out, seg_mask) hoặc (main_out, aux_out, seg_mask)
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    main_out = outputs[0]
                elif len(outputs) == 2:
                    main_out = outputs[0]
                else:
                    main_out = outputs[0]
            else:
                main_out = outputs

            probs = torch.softmax(main_out, dim=1)
            _, predicted = main_out.max(1)
            _, top5_preds = main_out.topk(5, dim=1)
            
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())
            all_top5_preds.append(top5_preds.cpu())
            
            # Đủ số lượng thì dừng
            total_collected = sum([img.shape[0] for img in all_images])
            if total_collected >= num_samples:
                break
    
    if not all_images:
        return
    
    # Gộp tất cả lại
    all_images_tensor = torch.cat(all_images, dim=0)[:num_samples]
    all_labels_tensor = torch.cat(all_labels, dim=0)[:num_samples]
    all_preds_tensor = torch.cat(all_preds, dim=0)[:num_samples]
    all_top5_tensor = torch.cat(all_top5_preds, dim=0)[:num_samples]
    
    # Giải phóng memory từ lists
    del all_images, all_labels, all_preds, all_top5_preds
    
    # Tạo grid với annotations
    try:
        import torchvision.utils as vutils
        
        # Denormalize để hiển thị
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalized = all_images_tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        # Vẽ predictions trực tiếp lên ảnh
        annotated_pil_images = draw_predictions_on_images(
            denormalized, all_labels_tensor, all_preds_tensor, all_top5_tensor,
            index_to_label=index_to_label, max_label_length=18
        )
        
        # Convert PIL images back to tensor
        annotated_tensors = []
        for pil_img in annotated_pil_images:
            img_np = np.array(pil_img)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            annotated_tensors.append(img_tensor)
            del img_np, img_tensor  # Giải phóng numpy array
        
        # Giải phóng PIL images
        del annotated_pil_images
        
        annotated_tensor_stack = torch.stack(annotated_tensors)
        del annotated_tensors
        
        # Tạo grid với ảnh đã được annotate
        grid = vutils.make_grid(annotated_tensor_stack, nrow=5, normalize=False, scale_each=False, pad_value=1.0)
        writer.add_image(f'{tag_prefix}/Predictions_Annotated', grid, epoch)
        del annotated_tensor_stack, grid
        
        # Vẫn giữ grid ảnh gốc (không có annotations)
        grid_original = vutils.make_grid(denormalized, nrow=5, normalize=False, scale_each=False)
        writer.add_image(f'{tag_prefix}/Random_Samples', grid_original, epoch)
        del denormalized, grid_original
        
        # Log thông tin Top-5 predictions (text log bổ sung)
        if index_to_label:
            top5_info = []
            for i in range(min(10, num_samples)):  # Log top 10 samples
                gt_label = index_to_label.get(all_labels_tensor[i].item(), f"Class_{all_labels_tensor[i].item()}")
                pred_label = index_to_label.get(all_preds_tensor[i].item(), f"Class_{all_preds_tensor[i].item()}")
                top5_labels = [index_to_label.get(all_top5_tensor[i][j].item(), f"Class_{all_top5_tensor[i][j].item()}") 
                              for j in range(5)]
                is_correct = "✓" if all_labels_tensor[i] == all_preds_tensor[i] else "✗"
                top5_info.append(f"Sample {i}: GT={gt_label}, Pred={pred_label} {is_correct}, Top5={top5_labels}")
            
            # Log dưới dạng text (bổ sung)
            info_text = "\n".join(top5_info)
            writer.add_text(f'{tag_prefix}/Top5_Predictions', info_text, epoch)
            
    except Exception as e:
        print(f"Lỗi visualize samples: {e}")
    except Exception as e:
        print(f"Lỗi visualize samples: {e}")


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Train 1 epoch với classification only (ResNet50 thuần).
    Loader trả về (images, labels) hoặc (images, labels, masks); mask không dùng.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        if images.nelement() == 0:
            continue

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if 5 > 1:
                top5_correct += top_k_accuracy(outputs, labels, k=5).item() * labels.size(0)

        pbar.set_postfix(
            loss=running_loss / (pbar.n + 1),
            acc=correct / total if total > 0 else 0,
        )

        if batch_idx % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    if total == 0:
        return 0, 0, 0

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    epoch_top5_acc = top5_correct / total if 5 > 1 else 0.0

    writer.add_scalar("Training/Loss", epoch_loss, epoch)
    writer.add_scalar("Training/Top-1 Accuracy", epoch_acc, epoch)
    if 5 > 1:
        writer.add_scalar("Training/Top-5 Accuracy", epoch_top5_acc, epoch)

    # Log gradients - chỉ log khi gradient tồn tại và không rỗng
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.numel() > 0:
            try:
                if not torch.isnan(param.grad).all() and not torch.isinf(param.grad).all():
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
            except (ValueError, RuntimeError):
                pass

    return epoch_loss, epoch_acc, epoch_top5_acc

def val_epoch(model, loader, criterion, device, epoch, tag_prefix='Validation', index_to_label=None, num_classes=None):
    """
    Validation epoch với đầy đủ tính năng: Top-5 predictions, visualization
    """
    # Ghi log ngay khi bắt đầu validation
    log_path = os.path.join(save_dir, 'train.log')
    try:
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(f"[VAL_START] Epoch {epoch} - Bắt đầu validation...\n")
            flog.flush()
    except:
        pass
    
    model.eval()
    running_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_top5_preds = []  # Lưu Top-5 predictions
    # Lưu lại một số batch để visualize
    sample_batch_images = None
    sample_batch_labels = None
    sample_batch_preds = None
    sample_batch_outputs = None
    
    pbar = tqdm(loader, desc=f"{tag_prefix} Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            if images.nelement() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            _, top5_preds = outputs.topk(5, dim=1)
            probs = torch.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if 5 > 1:
                top5_correct += top_k_accuracy(outputs, labels, k=5).item() * labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top5_preds.extend(top5_preds.cpu().numpy())

            if batch_idx == 0:
                sample_batch_images = images[:16].cpu().clone()
                sample_batch_labels = labels[:16].cpu().clone()
                sample_batch_preds = predicted[:16].cpu().clone()
                sample_batch_outputs = probs[:16].cpu().clone()
                sample_batch_top5 = top5_preds[:16].cpu().clone()
            
            del images, labels, outputs, predicted, top5_preds, probs
            
            pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=correct / total if total > 0 else 0)
            
            # Ghi log progress mỗi 50 batches và cleanup memory
            if (batch_idx + 1) % 50 == 0:
                try:
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"[VAL_PROGRESS] Epoch {epoch} - Batch {batch_idx+1}/{len(loader)}, processed={total}, acc={correct/total if total>0 else 0:.4f}\n")
                        flog.flush()
                except:
                    pass
                # Cleanup memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    pbar.close()
    
    # Ghi log sau khi hoàn thành validation loop
    try:
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(f"[VAL_LOOP_DONE] Epoch {epoch} - Đã xử lý {total} samples, bắt đầu visualization...\n")
            flog.flush()
    except:
        pass
    
    # Log ảnh mẫu với predictions được vẽ trực tiếp lên ảnh (mỗi epoch)
    if sample_batch_images is not None:
        try:
            import torchvision.utils as vutils
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denormalized = sample_batch_images * std + mean
            denormalized = torch.clamp(denormalized, 0, 1)
            
            # Vẽ predictions trực tiếp lên ảnh
            annotated_pil_images = draw_predictions_on_images(
                denormalized, sample_batch_labels, sample_batch_preds, sample_batch_top5,
                index_to_label=index_to_label, max_label_length=18
            )
            
            # Convert PIL images back to tensor
            annotated_tensors = []
            for pil_img in annotated_pil_images:
                img_np = np.array(pil_img)
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                annotated_tensors.append(img_tensor)
                del img_np, img_tensor  # Giải phóng numpy array
            
            # Giải phóng PIL images
            del annotated_pil_images
            
            annotated_tensor_stack = torch.stack(annotated_tensors)
            del annotated_tensors
            
            # Tạo grid với ảnh đã được annotate
            grid = vutils.make_grid(annotated_tensor_stack, nrow=4, normalize=False, scale_each=False, pad_value=1.0)
            writer.add_image(f'{tag_prefix}/Sample_Images_Annotated', grid, epoch)
            del annotated_tensor_stack, grid

            # Tạo visualization với ảnh + bảng 52 chữ cái + Top-5 confidences
            # Log từng ảnh riêng biệt (không gộp vào grid)
            num_samples_to_show = min(16, len(denormalized))
            for i in range(num_samples_to_show):
                try:
                    img_tensor = denormalized[i]
                    top1_idx = sample_batch_preds[i].item()
                    top1_font = index_to_label.get(top1_idx, f"Class_{top1_idx}") if index_to_label else f"Class_{top1_idx}"
                    gt_label = index_to_label.get(sample_batch_labels[i].item(), f"Class_{sample_batch_labels[i].item()}") if index_to_label else f"Class_{sample_batch_labels[i].item()}"
                    pred_label = index_to_label.get(top1_idx, f"Class_{top1_idx}") if index_to_label else f"Class_{top1_idx}"
                    
                    # Lấy Top-5 indices và probabilities
                    top5_indices = sample_batch_top5[i].numpy().tolist()
                    # Lấy probabilities từ softmax output
                    top5_probs = [sample_batch_outputs[i][idx].item() for idx in top5_indices]
                    
                    # Tạo visualization: ảnh + bảng 52 chữ cái bên phải
                    vis_img = create_prediction_visualization(
                        img_tensor, top1_font, top5_indices, top5_probs,
                        gt_label, pred_label, index_to_label,
                        font_dir=FONT_VAL_DIR, image_width=224, alphabet_width=224
                    )
                    
                    # Convert sang tensor và log từng ảnh riêng biệt
                    vis_tensor = torch.from_numpy(vis_img.transpose(2, 0, 1)).float() / 255.0
                    # Log từng ảnh với tag riêng để hiển thị riêng biệt
                    writer.add_image(f'{tag_prefix}/Prediction_{i+1}_Image_Alphabet_Top5', vis_tensor, epoch)
                    del vis_tensor, vis_img  # Giải phóng memory
                    
                except Exception as e:
                    print(f"Lỗi tạo visualization cho sample {i}: {e}")
                    # Fallback: log ảnh gốc
                    writer.add_image(f'{tag_prefix}/Prediction_{i+1}_Image_Alphabet_Top5', denormalized[i], epoch)
            
            # Log Top-5 predictions với GT cho 10 samples đầu (text log bổ sung) - trước khi xóa biến
            if index_to_label and sample_batch_top5 is not None:
                top5_info = []
                for i in range(min(10, len(sample_batch_labels))):
                    gt_label = index_to_label.get(sample_batch_labels[i].item(), f"Class_{sample_batch_labels[i].item()}")
                    pred_label = index_to_label.get(sample_batch_preds[i].item(), f"Class_{sample_batch_preds[i].item()}")
                    top5_labels = [index_to_label.get(sample_batch_top5[i][j].item(), f"Class_{sample_batch_top5[i][j].item()}") 
                                  for j in range(5)]
                    is_correct = "✓" if sample_batch_labels[i] == sample_batch_preds[i] else "✗"
                    top5_info.append(f"Sample {i}: GT={gt_label}, Pred={pred_label} {is_correct}, Top5={top5_labels}")
                info_text = "\n".join(top5_info)
                writer.add_text(f'{tag_prefix}/Top5_Predictions_Sample', info_text, epoch)
                del top5_info, info_text
            
            # Vẫn giữ grid ảnh gốc (không có annotations)
            grid_original = vutils.make_grid(denormalized, nrow=4, normalize=False, scale_each=False)
            writer.add_image(f'{tag_prefix}/Sample_Images', grid_original, epoch)
            del denormalized, grid_original
            del sample_batch_images, sample_batch_labels, sample_batch_preds, sample_batch_outputs, sample_batch_top5
        except Exception as e:
            print(f"Lỗi log ảnh {tag_prefix}: {e}")
    
    # Visualize 20 random samples
    try:
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(f"[VAL_VISUALIZE] Epoch {epoch} - Bắt đầu visualize_random_samples...\n")
            flog.flush()
    except:
        pass
    visualize_random_samples(model, loader, device, writer, tag_prefix, epoch, num_samples=20, index_to_label=index_to_label)
    
    if total == 0:
        return 0, 0, 0, 0, 0
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    epoch_top5_acc = top5_correct / total if 5 > 1 else 0.0
    try:
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    except ValueError:
        f1_macro, f1_weighted = 0.0, 0.0

    # Log + print Seg IoU (nếu có GT mask và model trả seg_mask)
    writer.add_scalar(f'{tag_prefix}/Loss', epoch_loss, epoch)
    writer.add_scalar(f'{tag_prefix}/Top-1 Accuracy', epoch_acc, epoch)
    if 5 > 1:
        writer.add_scalar(f'{tag_prefix}/Top-5 Accuracy', epoch_top5_acc, epoch)
    writer.add_scalar(f'{tag_prefix}/F1 Macro', f1_macro, epoch)
    writer.add_scalar(f'{tag_prefix}/F1 Weighted', f1_weighted, epoch)

    # --- Per-font F1 distribution với transition tracking (mỗi 20 epoch) ---
    # Chỉ chạy cho validation (tag_prefix chứa 'Validation') để tránh tốn chi phí không cần thiết
    if (epoch % 20 == 0) and (num_classes is not None) and len(all_labels) > 0 and len(all_preds) > 0 and ('Validation' in tag_prefix):
        try:
            global _font_bin_history, _last_tracked_epoch, _bin_base_colors
            
            # F1 cho từng class (font)
            per_class_f1 = f1_score(
                all_labels,
                all_preds,
                average=None,
                labels=list(range(num_classes)),
                zero_division=0
            )

            # Các khoảng F1 theo phần trăm
            bins = [
                (0.0, 0.1),
                (0.1, 0.2),
                (0.2, 0.3),
                (0.3, 0.4),
                (0.4, 0.5),
                (0.5, 0.6),
                (0.6, 0.7),
                (0.7, 0.8),
                (0.8, 0.9),
                (0.9, 1.01),  # 1.01 để bao trùm cả 1.0
            ]
            bin_labels = [
                "0-10%",
                "10-20%",
                "20-30%",
                "30-40%",
                "40-50%",
                "50-60%",
                "60-70%",
                "70-80%",
                "80-90%",
                "90-100%",
            ]
            num_bins = len(bins)

            # Xác định bin hiện tại cho mỗi font
            current_bin_assignment = {}  # {class_idx: bin_idx}
            for class_idx, f1_val in enumerate(per_class_f1):
                for bin_idx, (low, high) in enumerate(bins):
                    if low <= f1_val < high:
                        current_bin_assignment[class_idx] = bin_idx
                        break

            # Tính counts hiện tại
            current_counts = [0] * num_bins
            for class_idx, bin_idx in current_bin_assignment.items():
                current_counts[bin_idx] += 1

            # Tính transition matrix nếu có epoch trước
            transition_matrix = None  # transition_matrix[from_bin][to_bin] = số font chuyển từ from_bin sang to_bin
            previous_bin_assignment = None
            
            if _last_tracked_epoch >= 0 and _last_tracked_epoch in _font_bin_history:
                previous_bin_assignment = _font_bin_history[_last_tracked_epoch]
                transition_matrix = [[0] * num_bins for _ in range(num_bins)]
                
                # Tính transitions: font nào chuyển từ bin nào sang bin nào
                for class_idx in range(num_classes):
                    if class_idx in previous_bin_assignment and class_idx in current_bin_assignment:
                        from_bin = previous_bin_assignment[class_idx]
                        to_bin = current_bin_assignment[class_idx]
                        if from_bin != to_bin:
                            transition_matrix[from_bin][to_bin] += 1

            # Lưu bin assignment hiện tại
            _font_bin_history[epoch] = current_bin_assignment.copy()
            _last_tracked_epoch = epoch

            # Vẽ bar chart với transition colors
            fig, ax = plt.subplots(figsize=(12, 7))
            x = np.arange(len(bin_labels))
            
            if transition_matrix is not None and previous_bin_assignment is not None:
                # Tính previous counts
                previous_counts = [0] * num_bins
                for class_idx, bin_idx in previous_bin_assignment.items():
                    previous_counts[bin_idx] += 1
                
                # ===== PHẦN 1: Tính toán chính xác phần "giữ nguyên" và "chuyển dịch" =====
                # Với mỗi bin đích (to_bin):
                # - Phần "giữ nguyên" (stay): Font đã ở bin này ở epoch trước VÀ vẫn ở bin này
                # - Phần "chuyển vào" (moved in): Font từ bin khác chuyển vào bin này
                
                stay_heights = []  # Số font giữ nguyên trong mỗi bin (màu gốc của bin đích)
                moved_in_by_source = {}  # {to_bin: {from_bin: count}} - fonts chuyển vào từ bin nguồn nào
                
                for to_bin in range(num_bins):
                    # Tính số font giữ nguyên trong bin này
                    # = Số font đã ở bin này ở epoch trước - số font đã chuyển ĐI từ bin này
                    fonts_stayed = previous_counts[to_bin] - sum(
                        transition_matrix[to_bin][other_bin] 
                        for other_bin in range(num_bins) 
                        if other_bin != to_bin
                    )
                    stay_heights.append(max(0, fonts_stayed))
                    
                    # Tính số font chuyển VÀO bin này từ các bin khác
                    moved_in_by_source[to_bin] = {}
                    for from_bin in range(num_bins):
                        if from_bin != to_bin and transition_matrix[from_bin][to_bin] > 0:
                            moved_in_by_source[to_bin][from_bin] = transition_matrix[from_bin][to_bin]
                
                # ===== PHẦN 2: Vẽ stacked bars =====
                # Bước 1: Vẽ phần "giữ nguyên" (stay) - màu gốc của bin đích
                bottom = [0] * num_bins
                bars_stay = ax.bar(x, stay_heights, color=_bin_base_colors, 
                                   edgecolor='black', linewidth=1.5, alpha=0.9,
                                   label='Giữ nguyên (không chuyển bin)')
                bottom = stay_heights.copy()
                
                # Bước 2: Vẽ phần "chuyển vào" (moved in) - màu của bin nguồn
                # Vẽ theo thứ tự từ bin nguồn để dễ nhìn
                drawn_source_labels = set()
                for from_bin in range(num_bins):
                    for to_bin in range(num_bins):
                        if from_bin != to_bin and transition_matrix[from_bin][to_bin] > 0:
                            height = transition_matrix[from_bin][to_bin]
                            # Label chỉ hiển thị một lần cho mỗi bin nguồn
                            label = f'Chuyển từ {bin_labels[from_bin]}' if from_bin not in drawn_source_labels else ''
                            if label:
                                drawn_source_labels.add(from_bin)
                            
                            # Vẽ với màu của bin nguồn (from_bin)
                            ax.bar(x[to_bin], height, bottom=bottom[to_bin], 
                                  color=_bin_base_colors[from_bin], 
                                  edgecolor='black', linewidth=1.0,
                                  alpha=0.8, label=label, 
                                  hatch='///' if height > 0 else None)  # Hatch để phân biệt phần chuyển
                            bottom[to_bin] += height
                
                # Bước 3: Thêm text annotation cho tổng số font trong mỗi bin
                for to_bin in range(num_bins):
                    if current_counts[to_bin] > 0:
                        # Hiển thị tổng số font trong bin (stay + moved in)
                        ax.text(x[to_bin], bottom[to_bin] + max(current_counts) * 0.02, 
                               str(current_counts[to_bin]),
                               ha='center', va='bottom', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                        
                        # Hiển thị chi tiết: stay / moved in (nếu có)
                        if stay_heights[to_bin] > 0 and len(moved_in_by_source[to_bin]) > 0:
                            moved_total = sum(moved_in_by_source[to_bin].values())
                            detail_text = f'{int(stay_heights[to_bin])}+{int(moved_total)}'
                            ax.text(x[to_bin], bottom[to_bin] * 0.5, detail_text,
                                   ha='center', va='center', fontsize=8, 
                                   color='white', fontweight='bold')
                
                # Thêm legend để giải thích
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            else:
                # Epoch đầu tiên: chỉ vẽ màu gốc
                ax.bar(x, current_counts, color=_bin_base_colors, edgecolor='black', linewidth=1.5)
                # Thêm text annotation
                for i, count in enumerate(current_counts):
                    if count > 0:
                        ax.text(x[i], count + 0.5, str(count),
                               ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Số lượng font', fontsize=12, fontweight='bold')
            ax.set_xlabel('Khoảng F1-score (per-font)', fontsize=12, fontweight='bold')
            
            # Title với thông tin transition
            if transition_matrix is not None:
                total_transitions = sum(sum(row) for row in transition_matrix)
                title = f'Phân bố F1-score theo font - Epoch {epoch}\n'
                title += f'(Transition từ epoch {_last_tracked_epoch-20}: {total_transitions} fonts đã chuyển bin)'
                ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            else:
                ax.set_title(f'Phân bố F1-score theo font - Epoch {epoch} (Baseline)', 
                           fontsize=13, fontweight='bold', pad=15)
            
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_ylim(bottom=0, top=max(current_counts) * 1.2 if current_counts else 10)
            
            # Thêm text box giải thích màu sắc (nếu có transition)
            if transition_matrix is not None:
                explanation = "Màu sắc:\n• Phần đặc = Font giữ nguyên trong bin\n• Phần có hatch = Font chuyển từ bin khác (màu = bin nguồn)"
                ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       family='monospace')

            writer.add_figure(f'{tag_prefix}/PerFont_F1_Distribution', fig, global_step=epoch)
            plt.close(fig)
            
            # Log transition matrix summary
            if transition_matrix is not None:
                total_transitions = sum(sum(row) for row in transition_matrix)
                if total_transitions > 0:
                    print(f"  📊 F1 Bin Transitions (epoch {_last_tracked_epoch-20} → {epoch}): {total_transitions} fonts đã chuyển bin")
        except Exception as e:
            print(f"Lỗi tính/visualize per-font F1 distribution ở epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()

    # Log per-class F1 scores cho một số class quan trọng (mỗi 10 epoch để giảm overhead)
    if epoch % 10 == 0 and len(all_labels) > 0 and len(all_preds) > 0:
        try:
            from sklearn.metrics import f1_score as skl_f1_score
            unique_classes = sorted(set(all_labels))
            if len(unique_classes) <= 50 and index_to_label:  # Chỉ log nếu có ít hơn 50 classes
                per_class_f1 = skl_f1_score(all_labels, all_preds, average=None, zero_division=0)
                for class_idx, per_f1_val in enumerate(per_class_f1):
                    if class_idx < len(unique_classes):
                        class_name = index_to_label.get(unique_classes[class_idx], f"Class_{unique_classes[class_idx]}")
                        writer.add_scalar(f'Per_Class_F1/{tag_prefix}_{class_name}', per_f1_val, epoch)
        except Exception as e:
            print(f"Lỗi log per-class F1: {e}")
    
    # Ghi log khi hoàn thành validation
    try:
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(f"[VAL_DONE] Epoch {epoch} - val_loss={epoch_loss:.4f}, val_acc={epoch_acc:.4f}, val_top5_acc={epoch_top5_acc:.4f}\n")
            flog.flush()
    except:
        pass
    
    # Cleanup memory cuối cùng
    del all_preds, all_labels, all_top5_preds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return epoch_loss, epoch_acc, epoch_top5_acc, f1_macro, f1_weighted

def main():
    # Đảm bảo torch được nhận diện đúng trong hàm này
    import torch  # Re-import để đảm bảo scope đúng
    
    # --- 6. TẢI TÀI NGUYÊN VÀ TẠO DATASET/DATALOADER ---
    print("Đang tải tài nguyên...")
    allowed_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

    # Chỉ dùng chữ trắng nền đen: không load pattern/background
    pattern_paths_train = []
    background_paths_train = []
    pattern_paths_val = []
    background_paths_val = []
    print("[Train/Val] Chế độ 100% chữ trắng nền đen (không dùng pattern/background).")

    train_font_paths = list(paths.list_files(FONT_TRAIN_DIR, validExts=(".ttf", ".TTF", ".otf", ".OTF")))
    val_font_paths = list(paths.list_files(FONT_VAL_DIR, validExts=(".ttf", ".TTF", ".otf", ".OTF")))

    # Dùng folder làm class - tất cả các file font trong cùng folder có cùng label
    # Model sẽ học từ TẤT CẢ các file font trong folder (Regular, Bold, Italic, v.v.)
    all_font_families = set(get_font_family_label(p, FONT_TRAIN_DIR) for p in train_font_paths + val_font_paths)
    unique_labels = sorted(list(all_font_families))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    global NUM_CLASSES
    NUM_CLASSES = len(unique_labels)
    print(f"Phát hiện {NUM_CLASSES} font families (mỗi folder là một class).")
    print(f"Model sẽ học từ TẤT CẢ các file font trong mỗi folder (Regular, Bold, Italic, v.v.).")

    # Không còn load words.txt - tự động generate từ ngẫu nhiên
    print("Sử dụng tự động generate từ ngẫu nhiên (không dùng words.txt)")

    # Tạo dataset với channel_a_prob, augmentation và text generation ban đầu (sẽ được cập nhật mỗi epoch)
    initial_channel_a_prob = get_channel_a_prob_schedule(1, EPOCHS)
    initial_aug_params = get_augmentation_schedule(1, EPOCHS)
    initial_train_transform = build_train_transform(initial_aug_params)
    initial_text_params = get_text_generation_schedule(1, EPOCHS)
    train_dataset = OnTheFlyFontDataset(
        font_paths=train_font_paths,
        label_to_index=label_to_index,
        pattern_paths=pattern_paths_train,
        background_paths=background_paths_train,
        transform=initial_train_transform,
        samples_per_font=SAMPLES_PER_FONT,
        image_size=IMAGE_SHAPE,
        is_validation=False,
        words_list=None,  # Không cần words_list nữa
        channel_a_prob=initial_channel_a_prob,
        multiline_prob_2words=initial_text_params['multiline_prob_2words'],
        multiline_prob_letters=initial_text_params['multiline_prob_letters']
    )

    # Validation: không augment
    val_dataset = OnTheFlyFontDataset(
        font_paths=val_font_paths,
        label_to_index=label_to_index,
        pattern_paths=pattern_paths_val,
        background_paths=background_paths_val,
        transform=val_transform_no_aug,
        samples_per_font=SAMPLES_PER_FONT_VAL,
        image_size=IMAGE_SHAPE,
        is_validation=True,
        words_list=None
    )

    print(f"Dataset Training tạo với {len(train_dataset)} samples.")
    print(f"Dataset Validation tạo với {len(val_dataset)} samples (không augment).")

    # DataLoader – tối ưu để giảm bottleneck
    # Giảm prefetch_factor nếu có vấn đề memory, tăng nếu CPU đủ mạnh
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,  # Giảm từ 4 xuống 2 để tiết kiệm RAM
        persistent_workers=False,  # Tắt để giải phóng RAM giữa các epoch
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,  # Giảm từ 4 xuống 2 để tiết kiệm RAM
        persistent_workers=False,  # Tắt để giải phóng RAM giữa các epoch
        multiprocessing_context='spawn' if NUM_WORKERS > 0 else None
    )

    # --- 6. MODEL VÀ HUẤN LUYỆN ---
    print(f"Đang khởi tạo ResNet50 (classification only) với ImageNet weights (IMAGENET1K_V1)...")
    model = models.resnet50(weights='IMAGENET1K_V1')
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, NUM_CLASSES),
    )

    # Chọn checkpoint: BEST (chất lượng tốt nhất trên val, LR giảm) hoặc LAST (resume đúng tiến trình)
    best_checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_BEST.pth')
    last_checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_last.pth')
    checkpoint_loaded = False
    loaded_from_best = False

    def _load_checkpoint_into_model(cp_path, from_best_label):
        nonlocal checkpoint_loaded, loaded_from_best
        checkpoint_data = torch.load(cp_path, map_location=device)
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            state_dict = checkpoint_data['state_dict']
            checkpoint_num_classes = checkpoint_data.get('num_classes')
        else:
            state_dict = checkpoint_data
            checkpoint_num_classes = None
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            if 'aux_cls' in new_key or 'seg_head' in new_key:
                continue
            cleaned_state_dict[new_key] = v
        if 'fc.weight' in cleaned_state_dict and 'fc.7.weight' not in cleaned_state_dict:
            cleaned_state_dict['fc.7.weight'] = cleaned_state_dict.pop('fc.weight')
            cleaned_state_dict['fc.7.bias'] = cleaned_state_dict.pop('fc.bias')
        elif 'fc.1.weight' in cleaned_state_dict and 'fc.7.weight' not in cleaned_state_dict:
            cleaned_state_dict['fc.7.weight'] = cleaned_state_dict.pop('fc.1.weight')
            cleaned_state_dict['fc.7.bias'] = cleaned_state_dict.pop('fc.1.bias')
        if checkpoint_num_classes and checkpoint_num_classes != NUM_CLASSES:
            print(f"⚠️  Số classes thay đổi: {checkpoint_num_classes} → {NUM_CLASSES}")
            print(f"   → Loại bỏ fc.* khỏi checkpoint")
            drop_prefixes = ("fc.",)
            cleaned_state_dict = {k: v for k, v in cleaned_state_dict.items() if not any(k.startswith(prefix) for prefix in drop_prefixes)}
        if isinstance(model, nn.DataParallel):
            missing_keys, unexpected_keys = model.module.load_state_dict(cleaned_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys (sẽ dùng random init): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys (sẽ bỏ qua): {len(unexpected_keys)} keys")
        print(f"✓ Đã load model state từ {from_best_label} checkpoint để train tiếp")
        checkpoint_loaded = True
        loaded_from_best = from_best_label == "BEST"

    # Ưu tiên BEST khi RESUME_FROM_BEST (14k→50k); ngược lại dùng LAST để resume đúng tiến trình
    if RESUME_FROM_BEST and os.path.exists(best_checkpoint_path):
        print(f"\n📦 RESUME_FROM_BEST=True → Load từ BEST: {best_checkpoint_path}")
        print("   (LR sẽ dùng LEARNING_RATE_FROM_BEST để tránh quên.)")
        try:
            _load_checkpoint_into_model(best_checkpoint_path, "BEST")
        except Exception as e:
            print(f"⚠️  Không thể load BEST checkpoint: {e}")
            if os.path.exists(last_checkpoint_path):
                print("   Thử load LAST checkpoint...")
                try:
                    _load_checkpoint_into_model(last_checkpoint_path, "LAST")
                except Exception as e2:
                    print(f"⚠️  Không thể load LAST: {e2}")
    elif os.path.exists(last_checkpoint_path):
        print(f"\n📦 Tìm thấy LAST checkpoint: {last_checkpoint_path}")
        print("Đang load để resume training từ checkpoint...")
        try:
            _load_checkpoint_into_model(last_checkpoint_path, "LAST")
        except Exception as e:
            print(f"⚠️  Không thể load LAST checkpoint: {e}")
            print("   Sẽ sử dụng ImageNet pretrained weights...")
    
    # Nếu không có checkpoint, load ImageNet pretrained weights
    if not checkpoint_loaded:
        print(f"\n📦 Không có checkpoint. Sử dụng ImageNet pretrained weights (IMAGENET1K_V1)...")
        try:
            pretrained = models.resnet50(weights='IMAGENET1K_V1').state_dict()
            model_dict = model.state_dict()
            pretrained = {
                k: v
                for k, v in pretrained.items()
                if k in model_dict and not k.startswith('fc') and 'aux_cls' not in k and 'seg' not in k
            }
            model_dict.update(pretrained)
            model.load_state_dict(model_dict, strict=False)
            print("✓ Đã load backbone ResNet50 pretrained từ ImageNet (bỏ qua aux/seg/fc).")
        except Exception as e:
            print(f"⚠️  Không thể load pretrained ResNet50: {e}")

    # Freeze toàn bộ, chỉ train layer3 + layer4 + fc (giống cấu hình tham khảo)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    if hasattr(model, 'aux_cls'):
        for param in model.aux_cls.parameters():
            param.requires_grad = True
    if hasattr(model, 'seg_head'):
        for param in model.seg_head.parameters():
            param.requires_grad = True

    model = model.to(device).float()
    # Kiểm tra số lượng GPU - đảm bảo torch được nhận diện đúng
    try:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except NameError:
        # Nếu torch không được nhận diện, import lại
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Sử dụng {num_gpus} GPUs!")
        model = nn.DataParallel(model)

    # Class weights (inverse frequency) + label smoothing
    train_labels = [label for label, _ in train_dataset.samples_list]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    weights = torch.tensor(
        [total_samples / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    def get_params(model, name):
        if isinstance(model, nn.DataParallel):
            return getattr(model.module, name).parameters()
        return getattr(model, name).parameters()

    # Khi load từ BEST (14k→50k): dùng LR thấp hơn (1/2) cho mọi layer
    if loaded_from_best:
        lr_fc = LEARNING_RATE_FROM_BEST
        lr_layer3 = 0.00005 * 0.5
        lr_layer4 = 0.0001 * 0.5
        print(f"   Learning rate (từ BEST): fc={lr_fc}, layer3={lr_layer3}, layer4={lr_layer4}")
    else:
        lr_fc = LEARNING_RATE
        lr_layer3 = 0.00005
        lr_layer4 = 0.0001
    optimizer = optim.AdamW(
        [
            {"params": get_params(model, "fc"), "lr": lr_fc},
            {"params": get_params(model, "layer3"), "lr": lr_layer3},
            {"params": get_params(model, "layer4"), "lr": lr_layer4},
        ],
        betas=(0.9, 0.999),
    )

    # SCHEDULER ĐƠN GIẢN - CosineAnnealingLR, giảm dần đều không restart
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,  # Giảm dần trong suốt EPOCHS epochs
        eta_min=5e-7   # Learning rate tối thiểu
    )

    # --- 7. VÒNG LẶP HUẤN LUYỆN CHÍNH ---
    early_stopping = EarlyStopping(patience=PATIENCE)
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'train_top5_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': []
    }

    best_val_loss = float('inf')
    start_epoch = 1
    
    # Load metrics và start_epoch từ CSV nếu có (để resume training từ epoch đúng)
    if checkpoint_loaded:
        print(f"\n📊 Đang load metrics để resume training...")
        metrics_csv_path = os.path.join(save_dir, 'training_metrics_resnet50.csv')
        if os.path.exists(metrics_csv_path):
            try:
                metrics_df = pd.read_csv(metrics_csv_path)
                if not metrics_df.empty:
                    # Lấy epoch cuối cùng
                    start_epoch = int(metrics_df['epoch'].max()) + 1
                    # Load lại metrics
                    for col in metrics.keys():
                        if col in metrics_df.columns:
                            metrics[col] = metrics_df[col].tolist()
                    print(f"✓ Đã load metrics từ CSV. Tiếp tục từ epoch {start_epoch}")
                    
                    # Load best_val_loss từ metrics (chỉ dùng nếu file BEST còn tồn tại)
                    best_model_path = os.path.join(save_dir, 'deepfont_resnet50_model_BEST.pth')
                    if os.path.exists(best_model_path) and 'val_loss' in metrics_df.columns and len(metrics_df) > 0:
                        best_val_loss = metrics_df['val_loss'].min()
                        print(f"✓ Best validation loss: {best_val_loss:.4f}")
                    elif not os.path.exists(best_model_path):
                        best_val_loss = float('inf')
                        print(f"✓ File BEST không tồn tại → sẽ lưu best từ val_loss của các epoch tiếp theo")
            except Exception as e:
                print(f"⚠️  Không thể load metrics từ CSV: {e}")
                # Thử load từ train.log
                try:
                    log_path = os.path.join(save_dir, 'train.log')
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                start_epoch = len(lines) + 1
                                print(f"✓ Đã đếm {len(lines)} epochs từ train.log. Tiếp tục từ epoch {start_epoch}")
                except:
                    pass
    else:
        print(f"\n🆕 Bắt đầu training từ đầu với ImageNet pretrained weights...")
    
    print(f"\n🚀 Bắt đầu training với {NUM_CLASSES} font families (đã được detect ở trên).")
    print(f"📊 Sẽ train từ epoch {start_epoch} đến epoch {EPOCHS}")
    
    # Cập nhật scheduler để bắt đầu từ epoch đúng (nếu resume training)
    if start_epoch > 1:
        # Step scheduler đến epoch bắt đầu (trừ 1 vì step() được gọi sau mỗi epoch)
        for _ in range(start_epoch - 1):
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        print(f"✓ Đã cập nhật scheduler. Learning rate hiện tại: {current_lr:.8f}")
    else:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"✓ Learning rate ban đầu: {current_lr:.8f}")
    
    # Biến để track epoch hiện tại (dùng cho signal handler)
    current_epoch_tracking = {'epoch': start_epoch}
    
    # Signal handler để catch SIGTERM/SIGINT và ghi log
    def signal_handler(signum, frame):
        epoch = current_epoch_tracking['epoch']
        print(f"\n⚠️  Nhận signal {signum} ở epoch {epoch}")
        log_path = os.path.join(save_dir, 'train.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as flog:
                flog.write(f"\n⚠️  Process bị kill bởi signal {signum} ở epoch {epoch}\n")
                flog.flush()
        except:
            pass
        
        # Cleanup DataLoaders trước khi exit
        try:
            if NUM_WORKERS > 0:
                for loader in [train_loader, val_loader]:
                    try:
                        if hasattr(loader, '_workers'):
                            for w in loader._workers:
                                if w.is_alive():
                                    w.terminate()
                                    w.join(timeout=0.5)
                    except:
                        pass
        except:
            pass
        
        # Cố gắng lưu checkpoint
        try:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_last.pth')
            state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint_data = {
                'state_dict': state_dict_to_save,
                'index_to_label': index_to_label,
                'label_to_index': label_to_index,
                'num_classes': NUM_CLASSES,
                'epoch': 0  # Unknown epoch
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"✓ Đã lưu checkpoint (signal handler): {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  Không thể lưu checkpoint trong signal handler: {e}")
        sys.exit(1)
    
    # Đăng ký signal handler
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            current_epoch_tracking['epoch'] = epoch
            try:
                print(f"\n{'='*60}")
                print(f"🚀 Bắt đầu Epoch {epoch}/{EPOCHS}")
                print(f"{'='*60}")
                
                # Cập nhật CHANNEL_A_PROB theo schedule
                current_channel_a_prob = get_channel_a_prob_schedule(epoch, EPOCHS)
                train_dataset.channel_a_prob = current_channel_a_prob
                print(f"📊 CHANNEL_A_PROB (skeleton ratio): {current_channel_a_prob:.3f}")
                
                # Cập nhật augmentation theo schedule (curriculum augmentation)
                aug_params = get_augmentation_schedule(epoch, EPOCHS)
                train_dataset.transform = build_train_transform(aug_params)
                
                # Cập nhật text generation theo schedule (curriculum text)
                text_params = get_text_generation_schedule(epoch, EPOCHS)
                train_dataset.multiline_prob_2words = text_params['multiline_prob_2words']
                train_dataset.multiline_prob_letters = text_params['multiline_prob_letters']
                
                print(f"📊 Augmentation schedule:")
                print(f"   Rotate limit: {aug_params['rotate_limit']}° (augment chữ: patch JPEG/blur/downscale)")
                print(f"📊 Text generation schedule:")
                print(f"   Multiline prob (2 words): {text_params['multiline_prob_2words']:.2f}")
                print(f"   Multiline prob (letters): {text_params['multiline_prob_letters']:.2f}")
                
                log_path = os.path.join(save_dir, 'train.log')
                try:
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"\n[START] Epoch {epoch} - Bắt đầu training...\n")
                        flog.write(f"[SCHEDULE] CHANNEL_A_PROB = {current_channel_a_prob:.3f}\n")
                        flog.write(f"[SCHEDULE] Augmentation: Rotate={aug_params['rotate_limit']}° (chữ: patch JPEG/blur/downscale)\n")
                        flog.write(f"[SCHEDULE] Text generation: Multiline_2Words={text_params['multiline_prob_2words']:.3f}, "
                                   f"Multiline_Letters={text_params['multiline_prob_letters']:.3f}\n")
                        flog.flush()
                except:
                    pass
                
                train_loss, train_acc, train_top5_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
                
                # Ghi log ngay sau train_epoch
                try:
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"[TRAIN] Epoch {epoch} - train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_top5_acc={train_top5_acc:.4f}\n")
                        flog.flush()
                except:
                    pass
                
                # Visualize 20 random samples từ training set
                try:
                    visualize_random_samples(model, train_loader, device, writer, 'Training', epoch, num_samples=20, index_to_label=index_to_label)
                except Exception as e:
                    print(f"⚠️  Lỗi visualize random samples ở epoch {epoch}: {e}")
                
                # Validation (chỉ Validation_NoAug, đã bỏ Validation trùng)
                print(f"📊 Bắt đầu validation epoch {epoch}...")
                val_loss, val_acc, val_top5_acc, val_f1_macro, val_f1_weighted = val_epoch(
                    model, val_loader, criterion, device, epoch,
                    tag_prefix='Validation_NoAug', index_to_label=index_to_label, num_classes=NUM_CLASSES
                )
                
                # Ghi log ngay sau val_epoch (QUAN TRỌNG - trước khi làm các việc khác)
                try:
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"[VAL] Epoch {epoch} - val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_top5_acc={val_top5_acc:.4f}\n")
                        flog.flush()  # Flush ngay để đảm bảo ghi vào disk
                except:
                    pass
                
                # Visualize 52 characters cho top 5 predicted fonts (mỗi 5 epoch)
                if epoch % 5 == 0:
                    try:
                        model.eval()
                        with torch.no_grad():
                            # Lấy một batch từ validation (images, labels, masks)
                            sample_batch = next(iter(val_loader))
                            if len(sample_batch) == 3:
                                sample_images, sample_labels, _ = sample_batch
                            else:
                                sample_images, sample_labels = sample_batch
                            sample_images = sample_images[:5].to(device)  # Lấy 5 ảnh
                            sample_labels = sample_labels[:5]
                            outputs = model(sample_images)
                            _, top5_preds = outputs.topk(5, dim=1)
                            
                            # Tạo grid cho 52 characters của top-1 predicted font cho mỗi ảnh
                            alphabet_grids = []
                            for i in range(min(5, len(sample_images))):
                                top1_font = index_to_label.get(top5_preds[i][0].item(), f"Class_{top5_preds[i][0].item()}")
                                alphabet_img = create_alphabet_grid(top1_font, width=800, height=800, font_dir=FONT_VAL_DIR)
                                alphabet_grids.append(torch.from_numpy(alphabet_img).permute(2, 0, 1).float() / 255.0)
                            
                            if alphabet_grids:
                                import torchvision.utils as vutils
                                alphabet_tensor = torch.stack(alphabet_grids)
                                grid_alphabet = vutils.make_grid(alphabet_tensor, nrow=5, normalize=False, scale_each=False)
                                writer.add_image(f'Validation_NoAug/Top1_Font_Alphabets_52chars', grid_alphabet, epoch)
                                del alphabet_tensor, grid_alphabet, alphabet_grids  # Giải phóng memory
                    except Exception as e:
                        print(f"⚠️  Lỗi visualize 52 characters ở epoch {epoch}: {e}")
                
                scheduler.step()  # CosineAnnealingLR giảm dần đều
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                early_stopping(val_loss)
                metrics['epoch'].append(epoch)
                metrics['train_loss'].append(train_loss)
                metrics['train_acc'].append(train_acc)
                metrics['train_top5_acc'].append(train_top5_acc)
                metrics['val_loss'].append(val_loss)
                metrics['val_acc'].append(val_acc)
                metrics['val_top5_acc'].append(val_top5_acc)
                metrics['val_f1_macro'].append(val_f1_macro)
                metrics['val_f1_weighted'].append(val_f1_weighted)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Training/Learning Rate', current_lr, epoch)
                # --- Ghi train.log (ghi lại đầy đủ) ---
                log_path = os.path.join(save_dir, 'train.log')
                try:
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_top5_acc={train_top5_acc:.4f}, "
                                   f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_top5_acc={val_top5_acc:.4f}, "
                                   f"val_f1_macro={val_f1_macro:.4f}, val_f1_weighted={val_f1_weighted:.4f}, "
                                   f"lr={current_lr:.8f}\n")
                        flog.flush()  # Flush ngay để đảm bảo ghi vào disk
                except Exception as e:
                    print(f"⚠️  Lỗi ghi log ở epoch {epoch}: {e}")
                
                checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_last.pth')
                try:
                    # Đảm bảo thư mục tồn tại
                    os.makedirs(save_dir, exist_ok=True)
                    # Lưu state_dict kèm mapping để inference có thể load đúng
                    state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    checkpoint_data = {
                        'state_dict': state_dict_to_save,
                        'index_to_label': index_to_label,
                        'label_to_index': label_to_index,
                        'num_classes': NUM_CLASSES,
                        'epoch': epoch
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"✓ Đã lưu checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"⚠️  Lỗi save checkpoint ở epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                
                if is_best:
                    print(f"🎉 Best model! Epoch: {epoch}, Val Loss: {val_loss:.4f}. Saving...")
                    best_model_path = f'{save_dir}/deepfont_resnet50_model_BEST.pth'
                    try:
                        # Copy cả checkpoint data (bao gồm mapping) thay vì chỉ copy file
                        shutil.copyfile(checkpoint_path, best_model_path)
                    except Exception as e:
                        print(f"⚠️  Lỗi copy best model ở epoch {epoch}: {e}")
                
                try:
                    metrics_df = pd.DataFrame(metrics)
                    metrics_df.to_csv(f'{save_dir}/training_metrics_resnet50.csv', index=False)
                except Exception as e:
                    print(f"⚠️  Lỗi save metrics CSV ở epoch {epoch}: {e}")
                
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            except KeyboardInterrupt:
                print(f"\n⚠️  Training bị dừng bởi user ở epoch {epoch}")
                print("Đang lưu checkpoint cuối cùng...")
                try:
                    # Đảm bảo thư mục tồn tại
                    os.makedirs(save_dir, exist_ok=True)
                    checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_last.pth')
                    state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    checkpoint_data = {
                        'state_dict': state_dict_to_save,
                        'index_to_label': index_to_label,
                        'label_to_index': label_to_index,
                        'num_classes': NUM_CLASSES,
                        'epoch': epoch
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"✓ Đã lưu checkpoint (KeyboardInterrupt): {checkpoint_path}")
                    log_path = os.path.join(save_dir, 'train.log')
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"Training interrupted at epoch {epoch}\n")
                except:
                    pass
                # Cleanup DataLoaders
                try:
                    if NUM_WORKERS > 0:
                        for loader in [train_loader, val_loader]:
                            try:
                                if hasattr(loader, '_workers'):
                                    for w in loader._workers:
                                        if w.is_alive():
                                            w.terminate()
                                            w.join(timeout=0.5)
                            except:
                                pass
                except:
                    pass
                break
            except Exception as e:
                print(f"\n❌ LỖI NGHIÊM TRỌNG ở epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                # Ghi lỗi vào log
                try:
                    log_path = os.path.join(save_dir, 'train.log')
                    with open(log_path, 'a', encoding='utf-8') as flog:
                        flog.write(f"\n❌ LỖI ở epoch {epoch}: {str(e)}\n")
                        flog.write(f"Traceback:\n{traceback.format_exc()}\n")
                except:
                    pass
                # Vẫn cố gắng lưu checkpoint nếu có thể
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    checkpoint_path = os.path.join(save_dir, 'deepfont_resnet50_model_last.pth')
                    state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    checkpoint_data = {
                        'state_dict': state_dict_to_save,
                        'index_to_label': index_to_label,
                        'label_to_index': label_to_index,
                        'num_classes': NUM_CLASSES,
                        'epoch': epoch
                    }
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"✓ Đã lưu checkpoint cuối cùng trước khi dừng: {checkpoint_path}")
                except:
                    pass
                # Hỏi user có muốn tiếp tục không
                print(f"\n⚠️  Training đã dừng ở epoch {epoch} do lỗi.")
                print("Bạn có thể resume training từ checkpoint sau khi sửa lỗi.")
                break
    finally:
        # Cleanup DataLoaders trong finally block để đảm bảo luôn được gọi
        try:
            if NUM_WORKERS > 0:
                print("Đang cleanup DataLoader workers...")
                for loader in [train_loader, val_loader]:
                    try:
                        if hasattr(loader, '_workers'):
                            for w in loader._workers:
                                if w.is_alive():
                                    w.terminate()
                                    w.join(timeout=1.0)
                    except Exception as e:
                        pass  # Ignore cleanup errors
        except:
            pass

    # Lưu training metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{save_dir}/training_metrics_resnet50.csv', index=False)
    final_model_path = f'{save_dir}/deepfont_resnet50_model_final.pth'
    final_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    final_checkpoint_data = {
        'state_dict': final_model_state,
        'index_to_label': index_to_label,
        'label_to_index': label_to_index,
        'num_classes': NUM_CLASSES,
        'epoch': EPOCHS
    }
    torch.save(final_checkpoint_data, final_model_path)

    print("\n📊 Creating final training summary plot...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        if metrics['epoch']:
            axes[0, 0].plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', color='blue')
            axes[0, 0].plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', color='red')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 1].plot(metrics['epoch'], metrics['train_acc'], label='Train Acc', color='blue')
            axes[0, 1].plot(metrics['epoch'], metrics['val_acc'], label='Val Acc', color='red')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            axes[1, 0].plot(metrics['epoch'], metrics['val_f1_macro'], label='F1 Macro', color='green')
            axes[1, 0].plot(metrics['epoch'], metrics['val_f1_weighted'], label='F1 Weighted', color='orange')
            axes[1, 0].set_title('Validation F1 Scores')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 1].plot(metrics['epoch'], metrics['train_top5_acc'], label='Train Top-5', color='blue')
            axes[1, 1].plot(metrics['epoch'], metrics['val_top5_acc'], label='Val Top-5', color='red')
            axes[1, 1].set_title('Top-5 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-5 Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            plt.tight_layout()
            summary_plot_path = f'{save_dir}/training_summary.png'
            plt.savefig(summary_plot_path, dpi=300)
            writer.add_figure('Final/Training_Summary', fig, 0)
            plt.close('all')  # Đóng tất cả figures để giải phóng memory
            print(f"   Training summary plots saved to: {summary_plot_path}")
        else:
            print("Không có dữ liệu metrics.")
            plt.close('all')  # Đóng tất cả figures để giải phóng memory
    except Exception as plot_err:
        print(f"Error creating summary plot: {plot_err}")

    # Cleanup DataLoaders để tránh lỗi multiprocessing khi shutdown
    try:
        if hasattr(train_loader, '_iterator'):
            del train_loader._iterator
        if hasattr(val_loader, '_iterator'):
            del val_loader._iterator
        # Force cleanup workers
        if NUM_WORKERS > 0:
            import torch.utils.data
            # Shutdown workers manually
            for loader in [train_loader, val_loader]:
                try:
                    if hasattr(loader, '_workers'):
                        for w in loader._workers:
                            if w.is_alive():
                                w.terminate()
                                w.join(timeout=1.0)
                except:
                    pass
    except Exception as cleanup_err:
        print(f"⚠️  Warning: Error during DataLoader cleanup: {cleanup_err}")
    
    writer.close()
    print("\nTraining completed!")
    print(f"   Best model: {save_dir}/deepfont_resnet50_model_BEST.pth")
    print(f"   TensorBoard: {save_dir}/runs/")
    print(f"\nTo view TensorBoard, run:")
    print(f"   tensorboard --logdir={save_dir}/runs --port=6006")


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()