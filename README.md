

# CIFAR-100 Overlap Class Analysis

This repository provides a workflow for identifying and analyzing overlapping classes in the CIFAR-100 dataset using DINO features and a simple classifier.

## Workflow

1. **Overlap Detection**
   - Extract DINO features for all CIFAR-100 images
   - Compute class centroids and cosine similarities
   - Identify overlapping classes (high similarity)
   - Save images and metadata for these classes

2. **Classification & Analysis**
   - Create train/val/test splits for overlapping classes
   - Train a simple MLP classifier
   - Generate classification report and confusion matrix
   - Analyze significant confusion pairs
   - Visualize results (UMAP, confusion matrix, sample images)

## Confusion Pair Analysis

Significant confusion pairs are detected by thresholding the normalized confusion matrix:
$$T = \mu + k \cdot \sigma$$
Pairs with $C_{ij} \geq T$ are flagged for further review. This highlights systematic misclassifications between visually or semantically similar classes.

## Results & Visuals

## Classification Report
| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0      | 0.9500    | 0.9500 | 0.9500   | 100     |
| 1      | 0.6981    | 0.7400 | 0.7184   | 100     |
| 2      | 0.8750    | 0.8400 | 0.8571   | 100     |
| 3      | 0.7091    | 0.7800 | 0.7429   | 100     |
| 4      | 0.8776    | 0.8600 | 0.8687   | 100     |
| 5      | 0.9062    | 0.8700 | 0.8878   | 100     |
| 6      | 0.9167    | 0.8800 | 0.8980   | 100     |
| 7      | 0.5882    | 0.6000 | 0.5941   | 100     |
| 8      | 0.8370    | 0.7700 | 0.8021   | 100     |
| 9      | 0.8936    | 0.8400 | 0.8660   | 100     |
| 10     | 0.8519    | 0.9200 | 0.8846   | 100     |
| 11     | 0.8505    | 0.9100 | 0.8792   | 100     |
| 12     | 0.8485    | 0.8400 | 0.8442   | 100     |
| 13     | 0.8351    | 0.8100 | 0.8223   | 100     |
| 14     | 0.8144    | 0.7900 | 0.8020   | 100     |
| 15     | 0.7476    | 0.7700 | 0.7586   | 100     |
| 16     | 0.9175    | 0.8900 | 0.9036   | 100     |
| 17     | 0.7449    | 0.7300 | 0.7374   | 100     |
| 18     | 0.8495    | 0.7900 | 0.8187   | 100     |
| 19     | 0.8478    | 0.7800 | 0.8125   | 100     |
| 20     | 0.6538    | 0.5100 | 0.5730   | 100     |
| 21     | 0.8889    | 0.8800 | 0.8844   | 100     |
| 22     | 0.8247    | 0.8000 | 0.8122   | 100     |
| 23     | 0.8252    | 0.8500 | 0.8374   | 100     |
| 24     | 0.7757    | 0.8300 | 0.8019   | 100     |
| 25     | 0.7835    | 0.7600 | 0.7716   | 100     |
| 26     | 0.6857    | 0.7200 | 0.7024   | 100     |
| 27     | 0.6289    | 0.6100 | 0.6193   | 100     |
| 28     | 0.6095    | 0.6400 | 0.6244   | 100     |
| 29     | 0.5948    | 0.6900 | 0.6389   | 100     |
| 30     | 0.8411    | 0.9000 | 0.8696   | 100     |
| 31     | 0.6019    | 0.6200 | 0.6108   | 100     |
| 32     | 0.8462    | 0.8800 | 0.8627   | 100     |
| 33     | 0.9406    | 0.9500 | 0.9453   | 100     |
| 34     | 0.8152    | 0.7500 | 0.7812   | 100     |
| 35     | 0.7547    | 0.8000 | 0.7767   | 100     |
| 36     | 0.7905    | 0.8300 | 0.8098   | 100     |
| 37     | 0.8462    | 0.6600 | 0.7416   | 100     |
| 38     | 0.8137    | 0.8300 | 0.8218   | 100     |
| 39     | 0.8557    | 0.8300 | 0.8426   | 100     |
| 40     | 0.7573    | 0.7800 | 0.7685   | 100     |
| 41     | 0.8351    | 0.8100 | 0.8223   | 100     |
| 42     | 0.6495    | 0.6300 | 0.6396   | 100     |
| 43     | 0.7064    | 0.7700 | 0.7368   | 100     |
| 44     | 0.5766    | 0.6400 | 0.6066   | 100     |
| 45     | 0.8542    | 0.8200 | 0.8367   | 100     |
| 46     | 0.7843    | 0.8000 | 0.7921   | 100     |
| 47     | 0.8173    | 0.8500 | 0.8333   | 100     |
| 48     | 0.7982    | 0.8700 | 0.8325   | 100     |
| 49     | 0.7579    | 0.7200 | 0.7385   | 100     |
| 50     | 0.8495    | 0.7900 | 0.8187   | 100     |
| 51     | 0.8182    | 0.7200 | 0.7660   | 100     |
| 52     | 0.7000    | 0.7000 | 0.7000   | 100     |
| 53     | 0.6121    | 0.7100 | 0.6574   | 100     |
|        |           |        |          |         |
|accuracy|           |        | 0.7835   | 5400    |
|macro avg| 0.7861   | 0.7835 | 0.7838   | 5400    |
|weighted avg| 0.7861| 0.7835 | 0.7838   | 5400    |
*
## Confusion Pair Analysis Report

| True        | Pred     | MisRate  | Precision(Pred) | Recall(True) | F1(True) | F1(Pred) |
|-------------|----------|----------|-----------------|--------------|----------|----------|
| maple_tree  | oak_tree | 0.2400   | 0.595           | 0.610        | 0.619    | 0.639    |
| girl        | woman    | 0.1600   | 0.612           | 0.510        | 0.573    | 0.657    |
| girl        | baby     | 0.1400   | 0.698           | 0.510        | 0.573    | 0.718    |
| man         | woman    | 0.1400   | 0.612           | 0.720        | 0.702    | 0.657    |
| mouse       | shrew    | 0.1400   | 0.577           | 0.640        | 0.624    | 0.607    |

## UMAP Visualizations

![girl vs baby](https://imgur.com/a/Fmrp7lR)
![girl vs woman](https://imgur.com/6UDne63)
![man vs woman](https://imgur.com/a/BCvo1g3)
![maple vs oak](https://imgur.com/a/eqwe-yLiPWzn)
![mouse vs shrew](https://imgur.com/a/xtN1x2T)


## Sample Images
![Samples](https://imgur.com/a/qkogqfZ)

---

## Notebooks

- `Find_Overlap_Classes.ipynb`: Overlap detection & dataset creation
- `Filter_classes.ipynb`: Training, evaluation, and visualization

