# Dog Breed Image Classification (DenseNet121 — No Transfer Learning)

PyTorch project for fine-grained dog breed classification on a curated subset of the Stanford Dogs dataset.

**Important (course requirement):** this version trains **DenseNet121 from scratch** (no ImageNet pretraining / no transfer learning).
The model uses the DenseNet121 *architecture* from torchvision, but weights are randomly initialized and the whole network is trained.
To match the original notebook, the from-scratch model applies **Xavier initialization** to Conv/Linear layers.

## Notebook → Repo migration

Your notebook has two key reproducible data steps that we keep as scripts:

1) **Filter breeds by image count** directly from the Kaggle ZIP  
   (e.g. threshold like `195` images per breed → ~25 breeds)

2) **Split train/val/test** using `splitfolders` in ImageFolder format

Everything else (training loop, checkpoints, resume) is in `src/`.

## Setup
```bash
pip install -r requirements.txt
```

## Data workflow

### 1) Create curated subset from the Kaggle ZIP
```bash
python scripts/filter_subset_from_zip.py --zip_path /path/to/stanford-dogs-dataset.zip --out_dir filtered_dataset --threshold 195
```

### 2) Split train/val/test
```bash
python scripts/prepare_data.py --images_dir filtered_dataset/images/Images --out_dir data/split --seed 42
```

## Train (from scratch)
```bash
python -m src.train --data_dir data/split --out_dir runs/scratch --epochs 300
```

### Resume
```bash
python -m src.train --data_dir data/split --out_dir runs/scratch --resume runs/scratch/checkpoints/last.pt --epochs 300
```

### Optional (still no transfer learning): 2-phase “freezing trick”
```bash
python -m src.train --data_dir data/split --out_dir runs/scratch \
  --epochs 300 --phase1_epochs 210 --phase2_epochs 90 --freeze_first_n 4 --phase2_lr 1e-4
```

## Evaluate
```bash
python -m src.evaluate --data_dir data/split --checkpoint runs/scratch/checkpoints/best.pt
```
