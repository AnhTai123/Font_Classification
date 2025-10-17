## Font Classification – Overview

End-to-end pipeline for generating synthetic font datasets (with optional pattern-filled text), training a ResNet50 classifier, and running fast inference with consistent padding/resize.

### Project Structure
- `dataset_generation.py`: Generate dataset with backgrounds and optional pattern-filled text. Optimized batching and parallel chunks.
- `extend_dataset.py`: Extend `dataset_output_english` from a specific font onward (e.g., Inder) using the same generator.
- `fast_generate.py`: One-shot fast dataset build (optimized defaults; 200 images/font).
- `generate_more_train_data.py`: Add more rendered images for existing train fonts.
- `generate_specific_font.py`: Generate images for a single font family.
- `augment_train_fonts.py`: Albumentations-based augmentation for images in `dataset_train`.
- `check_patterns.py`: Validate and clean corrupted images in `patterns/`.
- `font_detectionction.py` / `detection_2.py`: Training scripts (ResNet50 + Albumentations).
- `inference.py`: Inference with padding to 320x320 then resize to 224x224; batch support; results saved to `result/`.

### Requirements
- Python 3.10+
- PyTorch + torchvision
- Albumentations, OpenCV, Pillow, numpy, pandas, tqdm, loguru, scikit-learn, imutils, matplotlib, tabulate

Install (example):
```bash
pip install -r requirements.txt  # if available
# or install key libs
pip install torch torchvision albumentations opencv-python pillow numpy pandas tqdm loguru scikit-learn imutils matplotlib tabulate
```

### Paths and Key Folders
- Fonts: `fonts/ofl`
- Backgrounds: `backgrounds/`
- Patterns (optional): `patterns/` (png/jpg/jpeg)
- Output dataset: `dataset_output_english/`
- Train/Val splits: `dataset_train`, `dataset_val`
- Inference results: `result/`

---

## Dataset Generation

The generator mixes normal text and pattern-filled text. It uses a background either from images or solid color. Font size range and image counts are configurable.

### Quick generation (optimized)
Creates 200 images per font across all available font families.
```bash
python fast_generate.py
```
Defaults in `fast_generate.py`:
- `num_images_per_font = 200`
- `patterns_folder = "patterns"`
- `num_workers = 16`

### Extend an existing dataset
Start after a specific font family (e.g., Inder) and add new families only.
```bash
python extend_dataset.py
```
Tune inside the script:
- `start_from_font = "Inder"`
- `num_images_per_font = 200`
- `num_workers`, `patterns_folder`

### Generator configuration (dataset_generation.py)
- Font size range (default): `min=64`, `max=144`
- Normal vs pattern ratio: default ~30% pattern, 70% normal
- Batching: batch size 20 to reduce I/O overhead
- Workers (chunks): default 16

If you deleted `dataset_output_english`, just rerun:
```bash
python fast_generate.py
```

### Validate/clean pattern images
If you see errors like “image file is truncated”, clean invalid patterns:
```bash
python check_patterns.py
```

---

## Train

Two example training scripts are provided:
- `font_detectionction.py`: Has `train_transform` and `val_transform`; validation resizes to 224x224.
- `detection_2.py`: Similar, includes sampler, metrics logging, etc.

Typical setup:
1) Split dataset into train/val
```bash
python split_dataset.py  # default ratios
```
2) Update script variables (`train_dataset_path`, `val_dataset_path`, etc.).
3) Run training (example):
```bash
python font_detectionction.py
```

Optional augmentation for `dataset_train` only:
```bash
python augment_train_fonts.py dataset_train --add 90 --start_from Charmonman
```

Add rendered images (not Albumentations) to train:
```bash
python generate_more_train_data.py dataset_train --add_per_font 90 \
  --fonts_path fonts/ofl --backgrounds_path backgrounds/ --text_source single_word \
  --textfile words_alpha.txt
``;

Generate for a specific family:
```bash
python generate_specific_font.py 42dotSans --count 100 --out dataset_output_english \
  --fonts_path fonts/ofl --backgrounds_path backgrounds/ --text_source single_word \
  --textfile words_alpha.txt
```

---

## Inference

`inference.py` uses padding to 320x320 (keep aspect), then resizes to 224x224 to match validation preprocessing. Results are saved to `result/` as a CSV and visualization PNGs.

Configure at top of the file:
```python
model_path = 'deepfont_resnet50_epoch_78.pth'
dataset_path = 'dataset_train_1'  # used to build label mapping
random_data_path = 'dataset_train_1/AbhayaLibre/0003_Bold_Feature;640.jpg'
result_folder = 'result'
```

Run:
```bash
python inference.py
```

Batch inference is automatic when you pass a folder in `random_data_path`.

---

## Performance Tips
- Increase `num_workers` in generation to utilize more CPU cores.
- Keep `patterns/` curated; remove corrupt images using `check_patterns.py`.
- Batch size for generation is set to 20 to reduce I/O operations.
- JPEG `quality=85` gives a good speed/size trade-off.
- For long runs, launch in background:
```bash
mkdir -p logs
nohup python fast_generate.py > logs/generate_$(date +%F_%H-%M-%S).log 2>&1 &
```

---

## Troubleshooting
- “image file is truncated”: run `python check_patterns.py` to remove bad pattern files; the generator also falls back to normal text if a pattern fails.
- Class count mismatch in inference: ensure `dataset_path` used for label mapping matches your trained classes.
- Slow generation: confirm `num_workers` and that `patterns/` and `backgrounds/` are on fast storage.

---

## License
Fonts, backgrounds, and patterns may have their own licenses. Ensure compliance when distributing datasets.

