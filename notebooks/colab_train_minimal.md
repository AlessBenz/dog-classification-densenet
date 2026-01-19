# Minimal Colab launcher (copy/paste)

> Tip: set `out_dir` to Google Drive so checkpoints persist.

### 1) Mount Drive (optional)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2) Clone repo & install
```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd <YOUR_REPO_FOLDER>
!pip install -r requirements.txt
```

### 3) Prepare data (mirrors the notebook)

**Option A — Kaggle ZIP already in Drive**
```bash
!python scripts/filter_subset_from_zip.py --zip_path "/content/drive/MyDrive/stanford-dogs-dataset.zip" --out_dir filtered_dataset --threshold 200
!python scripts/prepare_data.py --images_dir filtered_dataset/images/Images --out_dir data/split --seed 42 --train 0.8 --val 0.2 --test 0
```

### 4) Train (many epochs)
```bash
!python -m src.train --data_dir data/split --out_dir "/content/drive/MyDrive/dog_runs/scratch" --epochs 300 --batch_size 64 --lr 3e-4 --no_test
```

### 5) Resume
```bash
!python -m src.train --data_dir data/split --out_dir "/content/drive/MyDrive/dog_runs/scratch" --resume "/content/drive/MyDrive/dog_runs/scratch/checkpoints/last.pt" --no_test
```

### 6) Optional: 2-phase freezing trick (still no transfer learning)
```bash
!python -m src.train --data_dir data/split --out_dir "/content/drive/MyDrive/dog_runs/scratch" \
  --epochs 300 --no_test \
  --phase1_epochs 180 --phase2_epochs 120 --freeze_first_n 8 --phase2_lr 1e-4

```

### 7) Optional: transfer learning baseline
```bash
!python -m src.train --data_dir data/split --out_dir "/content/drive/MyDrive/dog_runs/imagenet_lastblock" \
  --epochs 50 --use_pretrained --fine_tune last_block --lr 1e-4 --no_test
```
