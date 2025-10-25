# cifar-100-dataset-cdl

This repository contains a Jupyter Notebook (`notebookk.ipynb`) and helper folders used to create a focused CIFAR-100 dataset variation based on DINO (self-supervised) feature similarity. The notebook extracts DINO features for CIFAR-100 images, identifies "overlapping" classes by cosine similarity of class centroids, and writes subsetted training and test images (with CSV metadata) to a new dataset folder structure.

## High-level overview

- Load CIFAR-100 (train + test). Two transforms are used:
  - `transform_basic` — simple ToTensor() used when saving images in their original format.
  - `transform_features` — resize/center-crop/normalize used to compute DINO features.
- Use a pre-trained DINO Vision Transformer (via `timm`) to extract the CLS token features for every image.
- Compute per-class centroids (mean feature vector) and a pairwise cosine similarity matrix between class centroids.
- Mark classes as "overlapping" when their centroid similarity exceeds a threshold (default 0.9).
- For overlapping classes, save original train images into `new_dataset/{DATASET}_{class_name}/` and test images into `new_dataset/test_set/{DATASET}_{class_name}/`.
- Save CSV metadata files describing every saved image (`metadata_train.csv` and `test_set/metadata_test.csv`).

## What the notebook does (step-by-step)

1. Imports and configuration
	- Detects device (CPU/GPU).
	- Creates output directories: `feature_extraction/features/`, `new_dataset/` and `new_dataset/test_set/`.
	- Sets model name (`vit_small_patch16_224.dino`), feature dimensions (384), batch size and a `SIMILARITY_THRESHOLD` (default 0.9).

2. Data loading and transforms
	- Defines `transform_basic` (for saving original images) and `transform_features` (for DINO inputs).
	- Loads CIFAR-100 train and test splits into `original_datasets` and creates a concatenated dataset for feature extraction.

3. DINO model setup
	- Loads the specified pre-trained DINO model with `timm` and wraps it in a small extractor that returns the CLS token feature vector.

4. Feature extraction
	- If not already present, extracts features for all images in the concatenated CIFAR-100 dataset using the DINO extractor and saves them to `feature_extraction/features/{DATASET}_X_all_dino.pt` and labels to `{DATASET}_y_all.pt`.

5. Overlap identification
	- Loads saved features and labels.
	- Computes centroids (mean feature per class).
	- Builds a cosine similarity matrix between centroids and flags class pairs exceeding the similarity threshold. The union of classes appearing in those pairs becomes the `overlapping_classes` set.

6. Saving training and test images
	- Iterates the CIFAR-100 train split and saves images whose label is in the overlapping set into `new_dataset/{DATASET}_{class_name}/`.
	- Iterates the CIFAR-100 test split and saves images into `new_dataset/test_set/{DATASET}_{class_name}/`.
	- Generates two CSV files with metadata about saved images:
	  - `new_dataset/metadata_train.csv`
	  - `new_dataset/test_set/metadata_test.csv`

7. Final summary 
	- The notebook prints counts and absolute paths of the new dataset directories and saved-image totals.


## Files and folders created

- `feature_extraction/features/{DATASET}_X_all_dino.pt` — Torch tensor of all extracted DINO features.
- `feature_extraction/features/{DATASET}_y_all.pt` — Torch tensor of labels corresponding to features.
- `new_dataset/{DATASET}_{class_name}/` — PNG images saved from the original training split for overlapping classes.
- `new_dataset/test_set/{DATASET}_{class_name}/` — PNG images saved from the original test split for overlapping classes.
- `new_dataset/metadata_train.csv` — metadata for training images saved.
- `new_dataset/test_set/metadata_test.csv` — metadata for test images saved.

Each metadata CSV contains rows with these columns (as saved by the notebook):
- `new_class_name` — the new folder name used for the class (e.g. `CIFAR100_apple`).
- `original_dataset` — dataset short name (e.g. `CIFAR100`).
- `original_class_name` — human-readable class label.
- `original_label_idx` — class numeric index (0..99).
- `original_index_in_split` — the original dataset index within the train/test split.
- `split` — `train` or `test`.
- `saved_path` — absolute or relative filesystem path to the PNG file written.

## Requirements / prerequisites

- Python 3.8+ (tested with recent PyTorch/timm versions).
- PyTorch and torchvision (GPU optional but recommended for feature extraction speed).
- timm (for the DINO model checkpoint), scikit-learn, pandas, tqdm, pillow, numpy, umap-learn (if you run UMAP visualization).

You can install the common dependencies (example):

```powershell
python -m pip install torch torchvision timm scikit-learn pandas tqdm pillow numpy umap-learn
```

Note: match the `torch` install to your CUDA version following the official instructions if you plan to use a GPU.

## How to run

1. Open `notebookk.ipynb` in Jupyter / VS Code and run the cells in order.
2. The notebook is already structured to skip expensive recomputation when feature files exist. If you re-run from scratch, delete the feature files in `feature_extraction/features/` to force re-extraction.

Quick outline of key cells:
- Imports & configuration — set device, paths and hyperparameters.
- Data loading — download/load CIFAR-100 and prepare transforms.
- DINO model setup — loads the pre-trained model via `timm`.
- Feature extraction — compute and save features/labels (large memory / GPU usage possible).
- Overlap identification — computes centroids and selects overlapping classes.
- Save subsets — writes images and metadata CSVs for train and test.
- Final summary — prints counts and paths.

## Configuration tips and caveats

- Similarity threshold: The default 0.9 is strict — lower it if you want more classes considered overlapping.
- Batch size and device: feature extraction uses a configurable `BATCH_SIZE_FEATURES` and will be much slower on CPU.
- Disk space: saving full PNGs for many classes will increase disk usage. Monitor free space when extracting large subsets.
- Determinism: The notebook uses deterministic centroid computation (mean of features) but ensure RNG seeds if you need exact reproducibility across runs.


## Next steps / extensions

- Filter images by per-sample similarity to a class centroid (instead of class-level centroids) to create tighter subsets.
- Use alternative backbones or embeddings (other DINO checkpoints, supervised backbones).
- Export a dataset manifest or create a PyTorch ImageFolder-friendly layout with a label mapping file.

## License & attribution

This repository and notebook are provided for research/educational purposes. The DINO model is used via `timm` and is subject to its licensing. CIFAR-100 is a standard academic dataset; check and follow its terms of use when redistributing derived datasets.

---

Generated from `notebookk.ipynb` markdown cells and code comments.
