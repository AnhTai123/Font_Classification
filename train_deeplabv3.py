import os
import sys
import time
import argparse
import random
from dataclasses import dataclass

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import transforms as T
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -----------------------
# Quick knobs (easy to edit)
# -----------------------
# Bạn có thể đổi 2 dòng này rồi chạy: python3 train_deeplabv3.py
DEFAULT_EPOCHS = 800
DEFAULT_BATCH_SIZE = 64
# Tăng rotate augmentation: limit từ schedule + ROTATE_BOOST (tối đa ROTATE_MAX độ)
ROTATE_BOOST = 5
ROTATE_MAX = 15


# -----------------------
# Metrics & losses
# -----------------------

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss cho binary segmentation.
    logits: [B, 1, H, W]
    target: [B, 1, H, W] (0/1)
    """
    prob = torch.sigmoid(logits).float()
    target = target.float()
    intersection = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    IoU (Jaccard) cho binary segmentation.
    logits: [B, 1, H, W]
    target: [B, 1, H, W] (0/1)
    """
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    gt = (target > 0.5).float()
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt - pred * gt).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean()


def ensure_mask_match_image(images: torch.Tensor, masks: torch.Tensor):
    """
    Resize đồng nhất: ảnh là chuẩn, mask resize về đúng kích thước ảnh.
    Tránh mask to hơn/nhỏ hơn chữ trên ảnh do lệch size. Dùng mode='nearest' để giữ mask binary.
    """
    if images.shape[-2:] == masks.shape[-2:]:
        return images, masks
    masks = F.interpolate(masks.float(), size=images.shape[-2:], mode="nearest")
    return images, masks


def build_val_no_aug_transform(t1):
    """
    Transform cho validation: CHỈ EnsureSize + Normalize + ToTensorV2, KHÔNG xoay / augment.
    Dùng riêng trong train_deeplabv3 để valid viz không bị xoay ngang/dọc.
    """
    return t1.A.Compose(
        [
            t1.EnsureSizeTransform(p=1.0),
            t1.A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            t1.ToTensorV2(),
        ]
    )


def apply_text_region_augmentations(image_np, mask_np, rng):
    """
    Augment lên toàn ảnh (cả background + chữ): JPEG, blur nhẹ, downscale→upscale.
    Dùng khi có background/pattern; mask_np không đổi, chỉ sửa image_np.
    Dùng cho simple pipeline (OnTheFlyFontSegDataset). Pipeline train1 dùng FullImageAugmentTransform.
    """
    if image_np is None:
        return image_np
    H, W = image_np.shape[:2]
    if H < 4 or W < 4:
        return image_np
    img = image_np.copy()
    if rng.random() < 0.45:
        quality = rng.randint(50, 74)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if buf is not None:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None:
                img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), 0.35)
    if rng.random() < 0.4:
        scale = rng.uniform(0.25, 0.5)
        sw, sh = max(2, int(W * scale)), max(2, int(H * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    image_np[:] = img
    return image_np


class FullImageAugmentTransform(A.DualTransform):
    """
    Augment toàn ảnh (JPEG compression, Gaussian blur, downscale-upscale) cho train_deeplabv3.
    Áp dụng lên toàn bộ ảnh (cả background + chữ), không chỉ vùng text.
    Mask không bị thay đổi (apply_to_mask trả về mask gốc).
    """
    def __init__(
        self,
        p_jpeg=0.45,
        quality_min=50,
        quality_max=74,
        p_blur=0.25,
        p_downscale=0.4,
        scale_min=0.25,
        scale_max=0.5,
        always_apply=False,
        p=1.0,
    ):
        super(FullImageAugmentTransform, self).__init__(always_apply=always_apply, p=p)
        self.p_jpeg = p_jpeg
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p_blur = p_blur
        self.p_downscale = p_downscale
        self.scale_min = scale_min
        self.scale_max = scale_max

    def apply(self, img, **params):
        if img is None:
            return img
        H, W = img.shape[:2]
        if H < 4 or W < 4:
            return img
        
        result = img.copy()
        
        # JPEG compression
        if np.random.random() < self.p_jpeg:
            quality = np.random.randint(self.quality_min, self.quality_max + 1)
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
            if buf is not None:
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if dec is not None:
                    result = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        
        # Gaussian blur
        if np.random.random() < self.p_blur:
            result = cv2.GaussianBlur(result, (3, 3), 0.35)
        
        # Downscale-upscale
        if np.random.random() < self.p_downscale:
            scale = np.random.uniform(self.scale_min, self.scale_max)
            sw, sh = max(2, int(W * scale)), max(2, int(H * scale))
            small = cv2.resize(result, (sw, sh), interpolation=cv2.INTER_AREA)
            result = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        
        return result

    def apply_to_mask(self, mask, **params):
        # Mask không bị thay đổi
        return mask

    def get_transform_init_args_names(self):
        return ("p_jpeg", "quality_min", "quality_max", "p_blur", "p_downscale", "scale_min", "scale_max")


def build_rotate_only_transform(t1, rotate_limit: int, p: float = 0.5, jpeg_quality_range=(85, 100), jpeg_p: float = 0.5):
    """
    Train transform: Rotate + (optional JPEG full image) + EnsureSize + Normalize + ToTensorV2.
    Dùng cho pipeline train1 khi muốn thêm JPEG toàn ảnh; thường dùng build_deeplab_train_transform (chỉ Rotate+Affine, JPEG trên chữ ở train_1).
    """
    return t1.A.Compose(
        [
            t1.EnsureSizeTransform(p=1.0),
            t1.A.Rotate(
                limit=int(rotate_limit),
                p=float(p),
                border_mode=cv2.BORDER_CONSTANT,
                border_value=(255, 255, 255),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            t1.A.ImageCompression(quality_range=jpeg_quality_range, p=jpeg_p),
            t1.A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            t1.ToTensorV2(),
        ]
    )


def build_deeplab_train_transform(t1, aug_params: dict, p_rotate: float = 0.2, p_affine: float = 0.2):
    """
    Train transform của train_deeplabv3 (augment toàn ảnh):
    FullImageAugment (JPEG/blur/downscale toàn ảnh) + EnsureSize + Rotate + Affine + Normalize + ToTensorV2.
    aug_params: từ t1.get_augmentation_schedule(epoch, total_epochs) (rotate_limit, affine_scale).
    """
    return t1.A.Compose(
        [
            FullImageAugmentTransform(p_jpeg=0.45, quality_min=50, quality_max=74, p_blur=0.25, p_downscale=0.4, scale_min=0.25, scale_max=0.5),
            t1.EnsureSizeTransform(p=1.0),
            t1.A.Rotate(
                limit=aug_params["rotate_limit"],
                p=p_rotate,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=(255, 255, 255),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            t1.A.Affine(
                scale=aug_params.get("affine_scale", (0.94, 1.02)),
                translate_percent=0,
                rotate=0,
                shear=0,
                cval=(255, 255, 255),
                mode=cv2.BORDER_CONSTANT,
                p=p_affine,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            t1.A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            t1.ToTensorV2(),
        ]
    )


def build_simple_seg_train_transform(rotate_limit: int = 15, p_rotate: float = 0.2):
    """
    Transform cho simple pipeline: Rotate (toàn ảnh) + Normalize + ToTensorV2.
    Augment trên chữ (JPEG/blur) áp trong OnTheFlyFontSegDataset qua apply_text_region_augmentations.
    """
    return A.Compose(
        [
            A.Rotate(
                limit=rotate_limit,
                p=p_rotate,
                border_mode=cv2.BORDER_CONSTANT,
                border_value=(255, 255, 255),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# -----------------------
# Fallback simple dataset (optional)
# -----------------------

@dataclass
class FontRenderConfig:
    image_size: int = 224
    min_font_size: int = 28
    max_font_size: int = 150
    max_stroke_width: int = 0  # 0 = chữ trơn, không viền
    min_chars: int = 3
    max_chars: int = 12


def _list_font_files(font_dir: str):
    font_files = []
    for root, _, files in os.walk(font_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".ttf") or lf.endswith(".otf"):
                font_files.append(os.path.join(root, f))
    font_files.sort()
    return font_files


class OnTheFlyFontSegDataset(Dataset):
    """
    Dataset segmentation-only đơn giản (fallback). Có augment: JPEG/blur trên chữ + Rotate toàn ảnh (khi seg_transform được truyền).
    Khuyến nghị dùng --pipeline train1 (default).
    """

    def __init__(self, font_dir: str, length: int, cfg: FontRenderConfig, seed: int = 1337, train: bool = True, seg_transform=None):
        from PIL import Image, ImageDraw, ImageFont

        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageFont = ImageFont

        self.font_files = _list_font_files(font_dir)
        if not self.font_files:
            raise RuntimeError(f"Không tìm thấy font .ttf/.otf trong: {font_dir}")
        self.length = int(length)
        self.cfg = cfg
        self.seed = int(seed)
        self.train = bool(train)
        self.seg_transform = seg_transform  # A.Compose (Rotate + Normalize + ToTensorV2) khi train
        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx * 9973)
        font_path = rng.choice(self.font_files)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        n = rng.randint(self.cfg.min_chars, self.cfg.max_chars)
        text = "".join(rng.choice(alphabet) for _ in range(n))

        font_size = rng.randint(self.cfg.min_font_size, self.cfg.max_font_size)
        stroke_width = 0  # Chữ trơn, không viền

        W = H = self.cfg.image_size
        img = self.Image.new("RGB", (W, H), (255, 255, 255))
        mask = self.Image.new("L", (W, H), 0)  # Cùng kích thước với ảnh

        try:
            font = self.ImageFont.truetype(font_path, size=font_size)
        except Exception:
            font = self.ImageFont.truetype(font_path, size=max(self.cfg.min_font_size, 28))

        draw_img = self.ImageDraw.Draw(img)
        draw_mask = self.ImageDraw.Draw(mask)

        try:
            bbox = font.getbbox(text, stroke_width=stroke_width)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = draw_img.textsize(text, font=font)

        tw = min(tw, W - 4)
        th = min(th, H - 4)
        x = rng.randint(2, max(2, W - tw - 2))
        y = rng.randint(2, max(2, H - th - 2))

        col = rng.randint(0, 40) if self.train else 0
        draw_img.text((x, y), text, fill=(col, col, col), font=font, stroke_width=0, stroke_fill=None)
        draw_mask.text((x, y), text, fill=255, font=font, stroke_width=0, stroke_fill=None)

        if self.train and self.seg_transform is not None:
            # Augment giống train_1: JPEG/blur trên chữ rồi Rotate toàn ảnh
            img_np = np.array(img)
            mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
            img_np = apply_text_region_augmentations(img_np, mask_np, rng)
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            out = self.seg_transform(image=img_np, mask=mask_uint8)
            img_t = out["image"]
            m = out["mask"]
            if isinstance(m, torch.Tensor):
                mask_t = (m.float() > 0.5).float().unsqueeze(0)
            else:
                mask_t = torch.from_numpy((np.array(m, dtype=np.float32) > 0.5).astype(np.float32)).unsqueeze(0)
            return img_t, mask_t

        if self.train:
            angle = rng.uniform(-2.0, 2.0)
            img = img.rotate(angle, resample=self.Image.BILINEAR, fillcolor=(255, 255, 255))
            mask = mask.rotate(angle, resample=self.Image.NEAREST, fillcolor=0)

        img_t = self.img_transform(img)
        mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        return img_t, mask_t


# -----------------------
# Adapter: reuse train_1.py pipeline (identical)
# -----------------------

def _list_images_recursive(dir_path: str, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")):
    paths = []
    if not dir_path or not os.path.isdir(dir_path):
        return paths
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


class Train1SegAdapter(Dataset):
    """
    Adapter để dùng pipeline sinh ảnh/mask y hệt `train_1.py` nhưng trả (image, mask) cho segmentation-only.
    """

    def __init__(self, train1_dataset: Dataset):
        self.ds = train1_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # train_1 dataset có thể trả:
        # - (img, label, mask) bình thường
        # - (img, -1) khi sample invalid (fallback)
        # Mục tiêu ở đây: chỉ lấy sample hợp lệ có mask.
        if isinstance(sample, (tuple, list)) and len(sample) == 3:
            img, label, mask = sample
            # Filter invalid
            try:
                if isinstance(label, (int, np.integer)) and int(label) == -1:
                    return None
                if torch.is_tensor(label) and label.numel() == 1 and int(label.item()) == -1:
                    return None
            except Exception:
                pass
            return img, mask

        if isinstance(sample, (tuple, list)) and len(sample) == 2:
            img, second = sample
            # Nếu second là label -1 -> invalid
            try:
                if isinstance(second, (int, np.integer)) and int(second) == -1:
                    return None
                if torch.is_tensor(second) and second.numel() == 1 and int(second.item()) == -1:
                    return None
            except Exception:
                return None
            # Không có mask -> bỏ
            return None

        return None


def seg_collate_fn(batch):
    """Collate (img, mask) -> (images, masks). Đảm bảo mask cùng kích thước với ảnh (resize mask nếu lệch)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        empty = torch.tensor([])
        return empty, empty
    out_images, out_masks = [], []
    for img, mask in batch:
        if img.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0).float(), size=img.shape[-2:], mode="nearest"
            ).squeeze(0)
        out_images.append(img)
        out_masks.append(mask)
    return torch.stack(out_images).float(), torch.stack(out_masks).float()


# -----------------------
# Model: DeepLabV3 / DeepLabV3+ -> 1-channel output (font mask only)
# -----------------------

def build_deeplabv3(num_output_channels: int = 1, pretrained: bool = True, backbone: str = "mobilenet", model_type: str = "deeplabv3plus") -> nn.Module:
    """
    DeepLabV3 hoặc DeepLabV3+ cho binary segmentation font mask.
    - model_type "deeplabv3plus": DeepLabV3+ với MobileNetV3 (segmentation_models_pytorch) - khuyến nghị
    - model_type "deeplabv3": DeepLabV3 với backbone mobilenet/resnet50 (torchvision)
    Output logits: [B, 1, H, W]
    """
    model_type = (model_type or "deeplabv3plus").lower()
    backbone = (backbone or "mobilenet").lower()

    if model_type in ("deeplabv3plus", "deeplabv3+", "plus"):
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError("DeepLabV3+ cần segmentation-models-pytorch. Chạy: pip install segmentation-models-pytorch")
        enc = "mobilenet_v3_large" if backbone in ("mobilenet", "mobilenetv3", "mbv3") else "resnet50"
        enc_weights = "imagenet" if pretrained else None
        try:
            model = smp.DeepLabV3Plus(
                encoder_name=enc,
                encoder_weights=enc_weights,
                in_channels=3,
                classes=num_output_channels,
            )
        except Exception:
            if enc == "mobilenet_v3_large":
                model = smp.DeepLabV3Plus(
                    encoder_name="mobilenet_v2",
                    encoder_weights=enc_weights,
                    in_channels=3,
                    classes=num_output_channels,
                )
            else:
                raise
        return model

    # DeepLabV3 (torchvision)
    if backbone in ("mobilenet", "mobilenetv3", "mobilenet_v3", "mbv3"):
        try:
            weights = (
                torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )
            model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        except Exception:
            model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
    elif backbone in ("resnet50", "resnet"):
        try:
            weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        except Exception:
            model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Use 'mobilenet' or 'resnet50'.")

    last = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(last.in_channels, num_output_channels, kernel_size=1)
    if hasattr(model, "aux_classifier"):
        model.aux_classifier = None
    return model


# -----------------------
# Visualization helpers
# -----------------------

def _denormalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


@torch.no_grad()
def log_seg_viz(writer: SummaryWriter, tag: str, images: torch.Tensor, logits_1ch: torch.Tensor, masks_1ch: torch.Tensor, step: int, num_samples: int = 1):
    """
    Log Input | Pred Mask | GT Mask | Input×Pred (chỉ chữ) | Input×GT (chỉ chữ).
    Nhân ảnh đầu vào với mask để giữ phần chung (foreground), hiển thị trên TensorBoard.
    """
    if images is None or logits_1ch is None or masks_1ch is None:
        return
    if images.nelement() == 0:
        return

    n = min(int(num_samples), images.size(0))
    tiles = []
    for i in range(n):
        img_raw = images[i].detach()
        try:
            if (img_raw.min() >= 0.0) and (img_raw.max() <= 1.0):
                img = img_raw.clamp(0, 1)
            else:
                img = _denormalize_imagenet(img_raw)
        except Exception:
            img = _denormalize_imagenet(img_raw)
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.dim() == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        log_i = logits_1ch[i].detach()
        gt_i = masks_1ch[i].detach()
        if log_i.dim() == 2:
            log_i = log_i.unsqueeze(0)
        if gt_i.dim() == 2:
            gt_i = gt_i.unsqueeze(0)
        H, W = img.shape[-2:]
        if log_i.shape[-2:] != (H, W):
            log_i = F.interpolate(log_i.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        if gt_i.shape[-2:] != (H, W):
            gt_i = F.interpolate(gt_i.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)
        # 2. Mask binary: sigmoid + threshold → 0 (nền) và 1 (chữ/foreground)
        pred_bin = (torch.sigmoid(log_i) > 0.5).float()
        gt_bin = (gt_i > 0.5).float()
        pred = pred_bin.repeat(3, 1, 1)
        gt = gt_bin.repeat(3, 1, 1)
        # 3. Ảnh nhân mask với ảnh đầu vào: giữ đúng màu chữ của ảnh, nền = 0 (đen)
        input_x_pred = img * pred_bin.repeat(3, 1, 1)
        input_x_gt = img * gt_bin.repeat(3, 1, 1)
        tiles.extend([img.cpu(), pred.cpu(), gt.cpu(), input_x_pred.cpu(), input_x_gt.cpu()])

    grid = torchvision.utils.make_grid(tiles, nrow=5, normalize=False, scale_each=False, padding=0)
    writer.add_image(tag, grid, step)


# -----------------------
# Checkpoint loading
# -----------------------

def load_latest_checkpoint(save_dir: str, device: torch.device):
    """
    Tìm và load checkpoint mới nhất từ save_dir.
    Ưu tiên: deeplabv3_last.pth > deeplabv3_best.pth
    
    Returns:
        dict với keys: 'checkpoint_path', 'epoch', 'model_state', 'optimizer_state', 
                       'scaler_state', 'best_val_iou', 'args', hoặc None nếu không tìm thấy
    """
    last_path = os.path.join(save_dir, "deeplabv3_last.pth")
    best_path = os.path.join(save_dir, "deeplabv3_best.pth")
    
    checkpoint_path = None
    if os.path.isfile(last_path):
        checkpoint_path = last_path
    elif os.path.isfile(best_path):
        checkpoint_path = best_path
    
    if checkpoint_path is None:
        return None
    
    try:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        result = {
            "checkpoint_path": checkpoint_path,
            "epoch": checkpoint.get("epoch", 0),
            "model_state": checkpoint.get("model"),
            "optimizer_state": checkpoint.get("optimizer"),
            "scaler_state": checkpoint.get("scaler"),
            "best_val_iou": checkpoint.get("best_val_iou", -1.0),
            "args": checkpoint.get("args"),
            "lr": checkpoint.get("lr"),  # LR tại thời điểm save (để resume đúng LR, không dùng args.lr ban đầu)
        }
        
        print(f"✓ Loaded checkpoint từ epoch {result['epoch']}, best_val_iou={result['best_val_iou']:.4f}")
        return result
    except Exception as e:
        print(f"⚠️  Không thể load checkpoint: {e}")
        return None


# -----------------------
# Eval / Train
# -----------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True, desc: str = "Validation"):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_iou_mask = 0.0
    total_dice_mask = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images, masks = batch
        if torch.is_tensor(images) and images.nelement() == 0:
            continue
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        images, masks = ensure_mask_match_image(images, masks)

        with autocast(enabled=torch.cuda.is_available() and amp):
            out = model(images)
            logits = out["out"] if isinstance(out, dict) else out
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            logits_mask = logits[:, 0:1]
            loss = bce(logits_mask, masks) + dice_loss_from_logits(logits_mask, masks)

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_iou_mask += float(iou_from_logits(logits_mask, masks).item()) * bs
        total_dice_mask += float((1.0 - dice_loss_from_logits(logits_mask, masks)).item()) * bs
        n += bs

        if n > 0 and hasattr(pbar, "set_postfix"):
            try:
                pbar.set_postfix(loss=float(total_loss / n), iou=float(total_iou_mask / n))
            except Exception:
                pass

    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_iou_mask / n, total_dice_mask / n


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, "runs", "deeplabv3_seg"))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # -----------------------
    # DATA PIPELINE
    # -----------------------
    if args.pipeline == "train1":
        # Import train_1 ở chế độ "không side-effect" (không tạo save_1, không print Save directory, không tạo writer)
        os.environ["TRAIN1_IMPORT_SAFE"] = "1"
        import train_1 as t1

        font_paths = _list_font_files(args.font_dir)
        if not font_paths:
            raise RuntimeError(f"Không tìm thấy fonts trong {args.font_dir}")

        # NOTE: label_to_index chỉ để tạo danh sách samples trong OnTheFlyFontDataset (chọn font families).
        # Script này KHÔNG train classification: loss/metric chỉ dùng masks.
        labels = [t1.get_font_family_label(p) for p in font_paths]
        uniq = sorted(set(labels))
        label_to_index = {name: i for i, name in enumerate(uniq)}

        pattern_paths = _list_images_recursive(args.pattern_dir)
        background_paths = _list_images_recursive(args.background_dir)
        if not pattern_paths or not background_paths:
            raise RuntimeError(
                f"Thiếu Pattern/Background. pattern={len(pattern_paths)} background={len(background_paths)} "
                f"(dirs: {args.pattern_dir}, {args.background_dir})"
            )

        # Init giống train_1 (epoch 1), tăng rotate lên chút
        initial_aug = t1.get_augmentation_schedule(1, args.epochs)
        initial_rotate = min(ROTATE_MAX, max(1, initial_aug["rotate_limit"] + ROTATE_BOOST))
        initial_transform = build_rotate_only_transform(t1, rotate_limit=initial_rotate, p=0.5)
        initial_channel_a = t1.get_channel_a_prob_schedule(1, args.epochs)
        text_params = t1.get_text_generation_schedule(1, args.epochs)

        train1_train_ds = t1.OnTheFlyFontDataset(
            font_paths=font_paths,
            label_to_index=label_to_index,
            pattern_paths=pattern_paths,
            background_paths=background_paths,
            transform=initial_transform,
            samples_per_font=args.samples_per_font,
            image_size=(args.image_size, args.image_size),
            is_validation=False,
            channel_a_prob=initial_channel_a,
            multiline_prob_2words=text_params["multiline_prob_2words"],
            multiline_prob_letters=text_params["multiline_prob_letters"],
            skip_text_region_augment=True,
        )

        # Validation: transform KHÔNG xoay (EnsureSize + Normalize + ToTensorV2) → valid viz không xoay ngang/dọc
        train1_val_ds = t1.OnTheFlyFontDataset(
            font_paths=font_paths,
            label_to_index=label_to_index,
            pattern_paths=pattern_paths,
            background_paths=background_paths,
            transform=build_val_no_aug_transform(t1),
            samples_per_font=args.samples_per_font_val,
            image_size=(args.image_size, args.image_size),
            is_validation=True,
            channel_a_prob=0.0,
            multiline_prob_2words=text_params["multiline_prob_2words"],
            multiline_prob_letters=text_params["multiline_prob_letters"],
            skip_text_region_augment=True,
        )

        train_ds = Train1SegAdapter(train1_train_ds)
        val_ds = Train1SegAdapter(train1_val_ds)
        # Collate thuần segmentation (images, masks)
        # Không dùng collate_fn của train_1.py vì nó expect (image, label, mask).
        collate_fn = seg_collate_fn
    else:
        cfg = FontRenderConfig(image_size=args.image_size)
        train_ds = OnTheFlyFontSegDataset(
            args.font_dir,
            args.train_length,
            cfg,
            seed=args.seed,
            train=True,
            seg_transform=build_simple_seg_train_transform(rotate_limit=min(ROTATE_MAX, 15), p_rotate=0.2),
        )
        val_ds = OnTheFlyFontSegDataset(args.font_dir, args.val_length, cfg, seed=args.seed + 999, train=False, seg_transform=None)
        collate_fn = None

    # num_workers=0 tránh đơ/deadlock: pipeline train1 (PIL + OpenCV + font) nặng trong __getitem__
    nw = max(0, args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, nw // 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
    )

    model_type = getattr(args, "model_type", "deeplabv3plus")
    model = build_deeplabv3(
        num_output_channels=1,
        pretrained=args.pretrained,
        backbone=args.backbone,
        model_type=model_type,
    ).to(device)
    print(f"Model: {model_type} (backbone={args.backbone})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available() and args.amp)
    bce = nn.BCEWithLogitsLoss()

    best_val_iou = -1.0
    start_epoch = 1
    global_step = 0

    if getattr(args, "no_resume", False):
        args.resume = False
    # Load checkpoint: nếu optimizer state không load được thì vẫn đặt LR từ checkpoint để tránh loss tăng / IoU giảm khi train tiếp (do dùng args.lr cao).
    if args.resume:
        ckpt = load_latest_checkpoint(args.save_dir, device)
        if ckpt is not None:
            # Load model state
            if ckpt["model_state"] is not None:
                try:
                    model.load_state_dict(ckpt["model_state"], strict=True)
                    print("✓ Loaded model state")
                except Exception as e:
                    print(f"⚠️  Lỗi load model state: {e}")
                    print("   Tiếp tục với model mới...")
            
            # Load optimizer state (gồm momentum/exp_avg của AdamW)
            if ckpt["optimizer_state"] is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                    lr_restored = optimizer.param_groups[0]["lr"]
                    # Ưu tiên LR đã lưu trong checkpoint (tại epoch đó), tránh resume lại dùng args.lr ban đầu
                    if ckpt.get("lr") is not None:
                        lr_restored = ckpt["lr"]
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr_restored
                    print(f"✓ Loaded optimizer state (LR từ checkpoint = {lr_restored:.2e}, epoch {ckpt['epoch']})")
                except Exception as e:
                    print(f"⚠️  Lỗi load optimizer state: {e}")
            
            # Load scaler state
            if ckpt["scaler_state"] is not None:
                try:
                    scaler.load_state_dict(ckpt["scaler_state"])
                    print("✓ Loaded scaler state")
                except Exception as e:
                    print(f"⚠️  Lỗi load scaler state: {e}")
            
            # Resume từ epoch tiếp theo
            start_epoch = ckpt["epoch"] + 1
            best_val_iou = ckpt["best_val_iou"]
            print(f" Resuming từ epoch {start_epoch} (checkpoint epoch {ckpt['epoch']})")
        else:
            print("ℹ  Không tìm thấy checkpoint, bắt đầu training từ đầu")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Update schedule giống train_1.py
        if args.pipeline == "train1":
            os.environ["TRAIN1_IMPORT_SAFE"] = "1"
            import train_1 as t1
            base_ds = train_ds.ds
            base_ds.channel_a_prob = t1.get_channel_a_prob_schedule(epoch, args.epochs)
            aug_params = t1.get_augmentation_schedule(epoch, args.epochs)
            base_ds.transform = build_deeplab_train_transform(t1, aug_params)
            text_params = t1.get_text_generation_schedule(epoch, args.epochs)
            base_ds.multiline_prob_2words = text_params["multiline_prob_2words"]
            base_ds.multiline_prob_letters = text_params["multiline_prob_letters"]

        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            images, masks = batch
            if torch.is_tensor(images) and images.nelement() == 0:
                continue
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            images, masks = ensure_mask_match_image(images, masks)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available() and args.amp):
                out = model(images)
                logits = out["out"] if isinstance(out, dict) else out
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                logits_mask = logits[:, 0:1]
                loss = bce(logits_mask, masks) + dice_loss_from_logits(logits_mask, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            running_iou += float(iou_from_logits(logits_mask.detach(), masks).item()) * bs
            running_dice += float((1.0 - dice_loss_from_logits(logits_mask.detach(), masks)).item()) * bs
            n += bs
            global_step += 1

            if n > 0 and hasattr(pbar, "set_postfix"):
                try:
                    pbar.set_postfix(loss=float(running_loss / n), iou=float(running_iou / n))
                except Exception:
                    pass

            # Log sớm để bạn thấy “có chữ” ngay, và sau đó log theo chu kỳ viz_every
            if (global_step == 1) or (args.viz_every > 0 and (global_step % args.viz_every == 0)):
                try:
                    log_seg_viz(writer, "Train/Viz_Input_PredMask_GTMask", images, logits_mask, masks, global_step)
                except Exception:
                    pass

        train_loss = running_loss / max(1, n)
        train_iou = running_iou / max(1, n)
        train_dice = running_dice / max(1, n)

        val_loss, val_iou, val_dice = evaluate(
            model,
            val_loader,
            device,
            amp=args.amp,
            desc=f"Validation Epoch {epoch}",
        )

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} | "
            f"time={dt:.1f}s"
        )

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/IoU_Mask", train_iou, epoch)
        writer.add_scalar("Train/Dice_Mask", train_dice, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/IoU_Mask", val_iou, epoch)
        writer.add_scalar("Val/Dice_Mask", val_dice, epoch)

        if args.viz_val:
            try:
                # Mỗi epoch lấy batch khác nhau (round-robin) để ảnh valid viz đa dạng, không cố định 8 ảnh
                num_val_batches = len(val_loader)
                start_batch = (epoch - 1) % num_val_batches if num_val_batches else 0
                viz_images, viz_masks = [], []
                for bi, batch in enumerate(val_loader):
                    if bi < start_batch:
                        continue
                    if bi >= start_batch + args.viz_val_batches:
                        break
                    imgs, msk = batch
                    viz_images.append(imgs)
                    viz_masks.append(msk)
                if viz_images:
                    images = torch.cat(viz_images, dim=0).to(device, non_blocking=True)
                    masks = torch.cat(viz_masks, dim=0).to(device, non_blocking=True)
                    images, masks = ensure_mask_match_image(images, masks)
                    with autocast(enabled=torch.cuda.is_available() and args.amp):
                        out = model(images)
                        logits = out["out"] if isinstance(out, dict) else out
                        if logits.shape[-2:] != masks.shape[-2:]:
                            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    logits_mask = logits[:, 0:1]
                    log_seg_viz(writer, "Val/Viz_Input_PredMask_GTMask", images, logits_mask, masks, epoch, num_samples=args.viz_val_samples)
            except Exception as e:
                print(f"  [Val viz] Lỗi: {e}")

        # checkpoints
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            ckpt_path = os.path.join(args.save_dir, "deeplabv3_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "args": vars(args),
                    "lr": optimizer.param_groups[0]["lr"],  # LR tại epoch này, để resume đúng
                },
                ckpt_path,
            )
            print(f"✓ Saved best checkpoint: {ckpt_path} (val_iou={best_val_iou:.4f}, lr={optimizer.param_groups[0]['lr']:.2e})")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, "deeplabv3_last.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "args": vars(args),
                    "lr": optimizer.param_groups[0]["lr"],  # LR tại epoch này, để resume đúng
                },
                ckpt_path,
            )
            print(f"✓ Saved last checkpoint: {ckpt_path} (epoch={epoch}, lr={optimizer.param_groups[0]['lr']:.2e})")

    writer.close()
    print(f"Done. Best val IoU = {best_val_iou:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="DeepLabV3 font/text segmentation (binary) - train_1 pipeline compatible")
    p.add_argument("--font_dir", type=str, default="label", help="Folder chứa fonts (.ttf/.otf)")
    p.add_argument("--save_dir", type=str, default="save_deeplabv3plus_mobilenet", help="Folder lưu checkpoint/logs")
    p.add_argument("--pipeline", type=str, default="train1", choices=["train1", "simple"], help="Data pipeline to use")
    p.add_argument("--pattern_dir", type=str, default="Pattern", help="Pattern directory (train_1 pipeline)")
    p.add_argument("--background_dir", type=str, default="Background", help="Background directory (train_1 pipeline)")
    p.add_argument("--samples_per_font", type=int, default=10, help="Samples per font per epoch (train_1 pipeline)")
    p.add_argument("--samples_per_font_val", type=int, default=4, help="Val samples per font per epoch (train_1 pipeline)")
    p.add_argument("--val_no_aug", action="store_true", help="(Deprecated) Val luôn dùng val_transform_no_aug, không augment")

    # defaults: dễ chỉnh bằng DEFAULT_* ở đầu file
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8, help="0 = tránh đơ (khuyến nghị với pipeline train1). Tăng nếu dataset nhẹ.")
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction, help="Use mixed precision (CUDA only)")
    p.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction, help="Use pretrained DeepLabV3 weights")
    p.add_argument("--model_type", type=str, default="deeplabv3plus", choices=["deeplabv3", "deeplabv3plus"], help="deeplabv3plus = DeepLabV3+ MobileNet (khuyến nghị)")
    p.add_argument("--backbone", type=str, default="mobilenet", choices=["mobilenet", "resnet50"], help="Backbone (mobilenet cho deeplabv3plus)")
    p.add_argument("--device", type=str, default="", help="cuda/cpu, để trống = auto")

    # only used in fallback simple pipeline
    p.add_argument("--train_length", type=int, default=20000)
    p.add_argument("--val_length", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--viz_every", type=int, default=50, help="Log train viz every N steps (0=off). Thử 0 nếu train đơ sau một đoạn.")
    p.add_argument("--viz_val", default=True, action=argparse.BooleanOptionalAction, help="Log validation viz per epoch")
    p.add_argument("--viz_val_samples", type=int, default=8, help="Số mẫu validation hiển thị trên TensorBoard")
    p.add_argument("--viz_val_batches", type=int, default=2, help="Số batch val lấy để viz (đa dạng ảnh hơn)")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--resume", action="store_true", default=True, help="Resume từ checkpoint lần train trước nếu có (default: True)")
    p.add_argument("--no_resume", action="store_true", help="Không load checkpoint, train từ đầu (bỏ qua --resume)")
    p.add_argument("--resume_lr", type=float, default=None, help="Khi resume: đặt LR cố định (vd: 1e-4) để tránh LR quá cao làm loss tăng")
    p.add_argument("--resume_lr_scale", type=float, default=None, help="Khi resume: nhân LR từ checkpoint với số này (vd: 0.5 = giảm một nửa)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

