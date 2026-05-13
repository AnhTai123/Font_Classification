import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import cv2
import os
from imutils import paths
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==============================
# CẤU HÌNH
# ==============================
INPUT_FOLDER = "a"
OUTPUT_FOLDER = "result_a_1"
TOP_K = 5
MODEL_PATH = 'save_2/deepfont_resnet50_model_BEST.pth'
FONT_DIR = 'label'  # Cập nhật để dùng cấu trúc folder theo font family
IMAGE_SHAPE = (224, 224)

# ==============================
# 1. TẢI TÀI NGUYÊN
# ==============================
def get_font_family_label(font_path, font_base_dir=None):
    """
    Lấy tên folder chứa font file làm label (font family name).
    Giống hệt logic trong font_detectionction.py và font_classification.py
    
    Args:
        font_path: Đường dẫn đầy đủ đến file font
        font_base_dir: Thư mục gốc chứa fonts (ví dụ: 'label_2_organized'). 
                      Nếu None, sẽ tự động lấy tên folder chứa file.
    
    Returns:
        Tên folder (font family) làm label
    """
    # Lấy parent directory name
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
    # Với cấu trúc label_2_organized/Arial/Arial-Bold.ttf -> label = "Arial"
    return parent_dir_name

def build_class_mapping(font_dir):
    """
    Quét thư mục font để tạo lại ánh xạ (mapping) y hệt như lúc train.
    Sử dụng tên folder (font family) làm label thay vì tên file.
    """
    print(f"Đang quét {font_dir} để tạo lại ánh xạ class...")
    font_paths = list(paths.list_files(font_dir, validExts=(".ttf", ".TTF", ".otf", ".OTF")))
    if not font_paths:
        raise FileNotFoundError(f"Không tìm thấy file font nào trong: {font_dir}")
    
    # Lấy label từ tên folder (font family) thay vì tên file
    # Ví dụ: label_2_organized/Arial/Arial-Bold.ttf -> label = "Arial"
    all_font_families = set(get_font_family_label(p, font_dir) for p in font_paths)
    unique_labels = sorted(list(all_font_families))
    
    index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Tạo ánh xạ thành công. Tổng số {num_classes} font families (classes).")
    
    return index_to_label, num_classes


def define_model_architecture(num_classes):
    """
    Cấu trúc giống hệt train_1.py: ResNet50 + fc (Linear→ReLU→BatchNorm1d→Dropout→Linear→ReLU→BatchNorm1d→Linear).
    weights=None vì sẽ load từ checkpoint; train_1 dùng weights='IMAGENET1K_V1' lúc khởi tạo.
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
        
        # Xử lý format cũ (chỉ có state_dict) và format mới (có mapping)
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            # Format mới: có mapping trong checkpoint
            state_dict = checkpoint_data['state_dict']
            checkpoint_num_classes = checkpoint_data.get('num_classes')
            checkpoint_index_to_label = checkpoint_data.get('index_to_label')
            checkpoint_label_to_index = checkpoint_data.get('label_to_index')
        else:
            # Format cũ: chỉ có state_dict (backward compatible)
            state_dict = checkpoint_data
            checkpoint_num_classes = None
            checkpoint_index_to_label = None
            checkpoint_label_to_index = None
            print("⚠️  WARNING: Checkpoint không có mapping. Đây là checkpoint cũ.")
            print("   Khuyến nghị: Retrain model để có mapping trong checkpoint.")
        
        # Validate num_classes
        if checkpoint_num_classes is not None:
            if checkpoint_num_classes != num_classes_from_mapping:
                print(f"❌ LỖI NGHIÊM TRỌNG: num_classes không khớp!")
                print(f"   Checkpoint: {checkpoint_num_classes} classes")
                print(f"   Mapping hiện tại: {num_classes_from_mapping} classes")
                print(f"   Điều này sẽ dẫn đến prediction sai hoàn toàn!")
                sys.exit(1)
            print(f"✓ num_classes khớp: {checkpoint_num_classes} classes")
        
        # Khởi tạo model với num_classes từ checkpoint (nếu có) hoặc từ mapping
        model_num_classes = checkpoint_num_classes if checkpoint_num_classes is not None else num_classes_from_mapping
        model = define_model_architecture(model_num_classes)
        
        # Loại bỏ prefix "module." nếu có (từ DataParallel)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            cleaned_state_dict[new_key] = v
        
        # Checkpoint cũ (train_1 từng dùng Dropout) có fc.1.weight/bias. Map sang fc.weight/bias cho model hiện tại (fc = Linear).
        if 'fc.1.weight' in cleaned_state_dict and 'fc.weight' not in cleaned_state_dict:
            cleaned_state_dict['fc.weight'] = cleaned_state_dict.pop('fc.1.weight')
            cleaned_state_dict['fc.bias'] = cleaned_state_dict.pop('fc.1.bias')
        
        # Load: strict=False để bỏ qua keys thừa (nếu checkpoint cũ có aux_cls/seg_head)
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  WARNING: Thiếu keys trong checkpoint (sẽ dùng random init): {len(missing_keys)} keys")
            # Chỉ hiển thị một số keys đầu tiên để không quá dài
            if len(missing_keys) <= 10:
                print(f"   Missing keys: {missing_keys}")
            else:
                print(f"   Missing keys (first 10): {list(missing_keys)[:10]}...")
        
        if unexpected_keys:
            print(f"⚠️  WARNING: Keys không mong đợi trong checkpoint (sẽ bỏ qua): {len(unexpected_keys)} keys")
            # Chỉ hiển thị một số keys đầu tiên để không quá dài
            if len(unexpected_keys) <= 10:
                print(f"   Unexpected keys: {unexpected_keys}")
            else:
                print(f"   Unexpected keys (first 10): {list(unexpected_keys)[:10]}...")
        
        print("✓ Đã load model state thành công")
        
    except RuntimeError as e:
        # RuntimeError từ load_state_dict với strict=True
        print(f"❌ LỖI: Không tải được model từ {model_path}")
        print(f"   Chi tiết: {e}")
        print(f"   Có thể do kiến trúc model không khớp với checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ LỖI: Không tải được model từ {model_path}")
        print(f"   Chi tiết: {e}")
        sys.exit(1)
    
    model = model.to(device).float()
    model.eval()
    print("Model sẵn sàng.")
    
    # Trả về model và mapping (nếu có từ checkpoint)
    return model, checkpoint_index_to_label, checkpoint_label_to_index


# ==============================
# 2. TIỀN XỬ LÝ (GIỐNG 20k.py)
# ==============================
def ensure_size(image, **kwargs):
    """Ensure image is exactly 224×224 (should already be, but double-check)."""
    target_h, target_w = IMAGE_SHAPE
    img = image.copy()
    h, w = img.shape[:2]
    
    # If already correct size, return as-is
    if h == target_h and w == target_w:
        return img
    
    # Otherwise, resize to target (shouldn't happen, but safety check). Padding màu đen.
    if h != target_h or w != target_w:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        temp = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # padding đen
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

# Val transform giống hệt trong 20k.py (dòng 994-998)
val_transform = A.Compose([
    EnsureSizeTransform(p=1.0),  # Safety check (should already be 224×224)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def preprocess_image(image_path, transform):
    """
    Tải và xử lý một ảnh duy nhất - giống pipeline trong 20k.py.
    Sử dụng EnsureSizeTransform để normalize về 224×224 trước khi normalize.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")
            
        # Convert BGR to RGB (keep color, no grayscale conversion)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform (EnsureSizeTransform sẽ normalize về 224×224, sau đó Normalize và ToTensorV2)
        # Giống hệt cách 20k.py xử lý validation images
        augmented = transform(image=image)
        image_tensor = augmented['image']
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        return None


def predict(model, tensor, device, index_to_label, top_k=5):
    """
    Predict probabilities từ model.
    
    Args:
        model: Model đã train
        tensor: Input image tensor
        device: Device (cuda/cpu)
        index_to_label: Mapping từ index sang label
        top_k: Số lượng predictions top
    
    Returns:
        List of (label, confidence_percentage) tuples
    """
    if tensor is None: 
        return None
    tensor = tensor.to(device)
    model.eval()  # Đảm bảo model ở eval mode (không training)
    with torch.no_grad():
        # Model output: inference mode trả về main_out (tensor), training mode trả về tuple
        outputs = model(tensor)
        
        # Nếu model trả về tuple (có thể xảy ra nếu model.training = True), chỉ lấy main output
        if isinstance(outputs, tuple):
            # Tuple có thể là (main_out, aux_out, seg_mask) hoặc (main_out, seg_mask)
            # Luôn lấy phần tử đầu tiên (main_out)
            outputs = outputs[0]
        
        # Apply softmax để có probabilities
        probs = torch.softmax(outputs, dim=1)[0]
        top_p, top_i = torch.topk(probs, top_k)
    
    return [(index_to_label[i.item()], p.item()*100) for i, p in zip(top_i, top_p)]


def get_font_path(font_name, font_dir=FONT_DIR):
    """
    Tìm đường dẫn font từ tên font (tên folder).
    Với cấu trúc label_2_organized/Arial/Arial-Bold.ttf, tìm file font trong folder font_name.
    Ưu tiên file Regular/Normal hoặc file có tên gần giống với tên folder.
    """
    # Tìm folder chứa font
    font_folder = os.path.join(font_dir, font_name)
    if not os.path.isdir(font_folder):
        # Fallback: thử tìm như file trực tiếp (cho trường hợp cũ)
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


# ==============================
# BẢNG 26 CHỮ CÁI A-Z ĐẸP, ĐỦ, CÁCH ĐỀU, KHÔNG DÍNH
# ==============================
def create_alphabet_grid(font_name, confidence=None, width=1000, height=1000):
    """
    Tạo bảng 52 chữ cái: A-Z + a-z
    TO RÕ, CÁCH ĐỀU, KHÔNG THỪA TRỐNG, DỊCH SÁT TRÁI
    """
    try:
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # 52 chữ cái: A-Z + a-z
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        cols = 13  # 13 chữ mỗi hàng
        rows = 4   # 4 hàng → 52 chữ
        cell_w = width // cols
        cell_h = height // rows

        # CHỮ SIÊU TO HƠN CHO DỄ NHÌN
        base_font_size = int(min(cell_w * 1.25, cell_h * 1.0))
        base_font_size = max(48, base_font_size)  # Tối thiểu 48 để rõ hơn

        font_path = get_font_path(font_name)
        default_font = ImageFont.load_default()

        # Không vẽ tiêu đề trực tiếp lên ảnh nữa, sẽ dùng matplotlib title
        # Chỉ để start_y bắt đầu từ đầu để có nhiều không gian cho chữ cái
        start_y = 10

        for idx, char in enumerate(chars):
            col = idx % cols
            row = idx // cols
            cx = col * cell_w + cell_w // 2
            cy = start_y + row * cell_h + cell_h // 2

            try:
                test_font = ImageFont.truetype(font_path, base_font_size) if font_path else default_font
                bbox = draw.textbbox((0, 0), char, font=test_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                tw, th = 50, 50
                test_font = default_font

            # Tự động scale nếu quá to
            scale = min(0.95 * cell_w / max(tw, 1), 0.95 * cell_h / max(th, 1))
            if scale < 1:
                new_size = max(30, int(base_font_size * scale))
                try:
                    test_font = ImageFont.truetype(font_path, new_size) if font_path else default_font
                except:
                    test_font = default_font

            try:
                bbox = draw.textbbox((0, 0), char, font=test_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except:
                tw, th = 50, 50

            x = cx - tw // 2
            y = cy - th // 2

            x = max(2, min(x, width - 2))
            y = max(start_y + 5, min(y, height - 5))

            draw.text((x, y), char, fill='#000000', font=test_font)

        return np.array(img)

    except Exception as e:
        print(f"Lỗi tạo bảng chữ: {e}")
        return np.ones((height, width, 3), dtype=np.uint8) * 255
def tensor_to_display_rgb(tensor):
    """
    Chuyển tensor đã normalize (ImageNet) [1,3,H,W] -> numpy RGB [H,W,3] để imshow.
    Denormalize theo mean/std ImageNet.
    """
    if tensor is None or tensor.nelement() == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    x = tensor.detach()
    if x.dim() == 4:
        x = x[0]  # [3, H, W]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1, 3, H, W]
    x = x.cpu().numpy()
    x = x * std + mean
    x = np.clip(x, 0, 1)
    x = (x[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    return x


# 4. HIỂN THỊ KẾT QUẢ ĐẸP (Ảnh trái + 5 bảng chữ cái phải)
# ==============================
def display_prediction_results(image_path, results, image_tensor):
    from matplotlib.gridspec import GridSpec

    # Hiển thị ảnh sau khi padding (224×224, đúng đầu vào model)
    display_img = tensor_to_display_rgb(image_tensor)
    h, w = display_img.shape[:2]

    # Tạo figure với GridSpec; dùng constrained_layout thay tight_layout để tránh warning
    fig = plt.figure(figsize=(32, 18), constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1.3, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.25)

    # Bên trái: Ảnh sau khi padding (đầu vào model)
    ax_image = fig.add_subplot(gs[0])
    ax_image.imshow(display_img)
    ax_image.set_title(f'Ảnh sau padding (đầu vào model)\n{w}×{h}', 
                       fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
    ax_image.axis('off')
    
    # Bên phải: 5 bảng chữ cái cho Top-5 fonts
    alphabet_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[i + 1])
        alphabet_axes.append(ax)
    
    # Tạo và hiển thị bảng chữ cái cho từng font Top-5
    for idx, (font_name, confidence) in enumerate(results):
        if idx >= 5:
            break
        
        ax = alphabet_axes[idx]
        ax.axis('off')
        
        # Tạo bảng chữ cái (không có tiêu đề trong ảnh)
        alphabet = create_alphabet_grid(font_name, confidence=None, width=700, height=1000)
        ax.imshow(alphabet)
        
        # Hiển thị tiêu đề rõ ràng bằng matplotlib (không bị che)
        title_text = f"#{idx + 1}: {font_name}\n({confidence:.2f}%)"
        ax.set_title(title_text, fontsize=13, fontweight='bold', 
                    pad=12, color='#2c3e50', 
                    bbox=dict(boxstyle="round,pad=1.0", facecolor="#ecf0f1", 
                             edgecolor='#34495e', linewidth=2, alpha=0.95))
    
    plt.suptitle('Kết Quả Dự Đoán Font - Top 5', 
                 fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_FOLDER, f"prediction_{base}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Đã lưu: {out_path}")

    try: 
        plt.show()
    except: 
        print("Đã lưu file (không hiển thị).")
    plt.close()


# ==============================
# 5. XỬ LÝ BATCH (ĐÃ SỬA LỖI)
# ==============================
def process_batch_folder(input_folder, output_folder, model, device, index_to_label, transform):
    """
    Process batch folder.
    
    Args:
        input_folder: Thư mục chứa ảnh input
        output_folder: Thư mục lưu kết quả
        model: Model đã train
        device: Device (cuda/cpu)
        index_to_label: Mapping từ index sang label
        transform: Image transform
    """
    os.makedirs(output_folder, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    imgs = [p for ext in exts for p in paths.list_files(input_folder, validExts=(ext, ext.upper()))]

    if not imgs:
        print("Không tìm thấy ảnh!")
        return

    log_path = os.path.join(output_folder, "prediction_log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ DỰ ĐOÁN FONT\n" + "="*60 + "\n\n")

        for i, p in enumerate(imgs, 1):
            print(f"\n[{i}/{len(imgs)}] {os.path.basename(p)}")
            tensor = preprocess_image(p, transform)
            if tensor is None: continue  # ĐÃ SỬA: tensor is None
            results = predict(model, tensor, device, index_to_label, TOP_K)
            if not results: continue

            top1, conf1 = results[0]
            f.write(f"Ảnh: {os.path.basename(p)}\n")
            f.write(f"Top 1: {top1} ({conf1:.2f}%)\n")
            f.write("-"*40 + "\n")

            display_prediction_results(p, results, tensor)

    print(f"\nHOÀN THÀNH! Kết quả tại: {output_folder}")
    print(f"Log: {log_path}")


# ==============================
# 6. MAIN
# ==============================
if __name__ == "__main__":
    print(f"Khởi động inference từ: {INPUT_FOLDER}")
    if not os.path.exists(INPUT_FOLDER):
        print(f"LỖI: Không tìm thấy '{INPUT_FOLDER}'")
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Thử load mapping từ checkpoint trước
    try:
        checkpoint_data = torch.load(MODEL_PATH, map_location='cpu')
        if isinstance(checkpoint_data, dict) and 'index_to_label' in checkpoint_data:
            # Checkpoint mới: có mapping
            index_to_label = checkpoint_data['index_to_label']
            label_to_index = checkpoint_data.get('label_to_index')
            num_classes = checkpoint_data.get('num_classes')
            print(f"✓ Đã load mapping từ checkpoint: {num_classes} classes")
            print(f"   Mapping được load từ checkpoint (đảm bảo khớp 100% với training).")
            
            # Load model với mapping từ checkpoint
            model, _, _ = load_trained_model(MODEL_PATH, num_classes, device)
        else:
            # Checkpoint cũ: không có mapping, phải rebuild
            print("⚠️  Checkpoint cũ không có mapping. Đang rebuild mapping từ FONT_DIR...")
            print("   Lưu ý: Mapping rebuild có thể KHÔNG khớp với training!")
            index_to_label, num_classes = build_class_mapping(FONT_DIR)
            model, _, _ = load_trained_model(MODEL_PATH, num_classes, device)
    except Exception as e:
        print(f"⚠️  Không thể load checkpoint để kiểm tra mapping: {e}")
        print("   Đang rebuild mapping từ FONT_DIR...")
        print("   Lưu ý: Mapping rebuild có thể KHÔNG khớp với training!")
        index_to_label, num_classes = build_class_mapping(FONT_DIR)
        model, _, _ = load_trained_model(MODEL_PATH, num_classes, device)
    
    process_batch_folder(INPUT_FOLDER, OUTPUT_FOLDER, model, device, index_to_label, val_transform)
