# Minimal Colab launcher (copy/paste)

### 1) Mount Drive
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

### 3) Filter subset + split
```bash
!python scripts/filter_subset_from_zip.py --zip_path "/content/drive/MyDrive/stanford-dogs-dataset.zip" --out_dir filtered_dataset --threshold 195
!python scripts/prepare_data.py --images_dir filtered_dataset/images/Images --out_dir data/split --seed 42
```

### 4) Train 300 epochs with auto-resume
```bash
OUTDIR="/content/drive/MyDrive/dog_runs/scratch_300"
CKPT="$OUTDIR/checkpoints/last.pt"

if [ -f "$CKPT" ]; then
  python -m src.train --data_dir data/split --out_dir "$OUTDIR" --epochs 300 --resume "$CKPT"
else
  python -m src.train --data_dir data/split --out_dir "$OUTDIR" --epochs 300
fi
```
