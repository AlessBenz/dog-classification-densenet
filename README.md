# Dog Breed Image Classification (DenseNet121 — No Transfer Learning)

PyTorch project for fine-grained dog breed classification on a curated subset of the Stanford Dogs dataset.

**Important (course requirement):** this version trains **DenseNet121 from scratch** (no ImageNet pretraining / no transfer learning).
The model uses the DenseNet121 *architecture* from torchvision, but weights are randomly initialized and the whole network is trained.
To match the notebook, the from-scratch model applies **Xavier initialization** to Conv/Linear layers.

## Notebook → Repo migration (next step)

Your notebook has two key *reproducible* steps that we keep as scripts:

1) **Filter breeds by image count** directly from the Kaggle ZIP  
   (your notebook used a threshold like `195` images per breed to end up with ~25 breeds)

2) **Split train/val[/test]** in ImageFolder format (pure-Python, no extra deps)

Everything else (training loop, checkpoints, resume) is now in `src/`.

## Project structure

```
.
├── configs/
│   └── train.yaml
├── scripts/
│   ├── filter_subset_from_zip.py
│   └── prepare_data.py
├── src/
│   ├── data.py
│   ├── models/
│   │   └── densenet.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── notebooks/
    └── colab_train_minimal.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data workflow (mirrors the notebook)

### 1) Download the Kaggle ZIP
Download the Stanford Dogs dataset ZIP from Kaggle.

### 2) Create the curated subset (filter by threshold)
This reproduces your notebook “filtered_dataset” creation:

```bash
python scripts/filter_subset_from_zip.py \
  --zip_path /path/to/stanford-dogs-dataset.zip \
  --out_dir filtered_dataset \
  --threshold 200
```

Output:

```
filtered_dataset/
└── images/Images/<breed folders>/*.jpg
```

### 3) Split into train/val (no test)
If you prefer to use **all available data for model selection** (train+val) and skip a test set:

```bash
python scripts/prepare_data.py \
  --images_dir filtered_dataset/images/Images \
  --out_dir data/split \
  --seed 42 \
  --train 0.8 --val 0.2 --test 0
```

Produces:

```
data/split/
├── train/<class folders>/*.jpg
└── val/<class folders>/*.jpg
```

## Train (from scratch)

```bash
python -m src.train --data_dir data/split --out_dir runs/scratch --epochs 300 --batch_size 64 --lr 3e-4 --no_test
```

### Resume
```bash
python -m src.train --data_dir data/split --out_dir runs/scratch --resume runs/scratch/checkpoints/last.pt --no_test
```

### Optional (still no transfer learning): 2-phase “freezing trick”
If you want to mimic the notebook idea of training all layers first and then freezing early feature blocks:

```bash
python -m src.train --data_dir data/split --out_dir runs/scratch \
  --epochs 300 --no_test \
  --phase1_epochs 180 --phase2_epochs 120 --freeze_first_n 8 --phase2_lr 1e-4
```

This does **not** use ImageNet weights; it only changes which layers are trainable in phase 2.

### Optional: class balancing sampler
```bash
python -m src.train --data_dir data/split --out_dir runs/scratch --balanced_sampler
```

## Evaluate
```bash
python -m src.evaluate --data_dir data/split --checkpoint runs/scratch/checkpoints/best.pt
```

## (Optional) Transfer learning baseline
If you want a **comparison run with ImageNet weights**:

```bash
python -m src.train --data_dir data/split --out_dir runs/imagenet_lastblock \
  --epochs 50 --use_pretrained --fine_tune last_block --lr 1e-4 --no_test
```

## Colab
See `notebooks/colab_train_minimal.md` for a minimal Colab “launcher”.
