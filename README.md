# Skin Lesion Classification using Deep Learning

This project classifies dermatoscopic skin lesion images from the HAM10000 dataset into seven diagnostic categories using classical baselines, custom convolutional neural networks, and transfer learning models.

The main goal is to compare different modeling approaches and identify which model performs best for imbalanced medical image classification.

## Project Overview

Skin lesion classification is an important computer vision problem in medical imaging. Many skin cancer datasets are highly imbalanced, where common classes have many more samples than rare but clinically important classes. This project addresses that challenge using data augmentation, weighted sampling, class-weighted loss, and transfer learning.

The notebook trains and evaluates multiple models on the HAM10000 dataset and compares their performance using balanced accuracy, macro F1 score, AUC-ROC, confusion matrix analysis, and per-class recall.

## Dataset

The project uses the HAM10000 dataset, also known as Human Against Machine with 10000 training images.

Dataset details:

| Item | Description |
|---|---|
| Dataset | HAM10000 Skin Lesion Dataset |
| Total Images | 10,015 dermoscopy images |
| Image Format | JPG |
| Metadata | CSV file with diagnosis, patient details, lesion ID, age, sex, and localization |
| Number of Classes | 7 |
| Source | Kaggle dataset: `kmader/skin-cancer-mnist-ham10000` |

## Lesion Classes

| Code | Class Name |
|---|---|
| `mel` | Melanoma |
| `nv` | Melanocytic Nevus |
| `bcc` | Basal Cell Carcinoma |
| `akiec` | Actinic Keratosis |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `vasc` | Vascular Lesion |

## Problem Statement

The objective is to build a deep learning pipeline that can classify skin lesion images into one of seven categories.

The major challenges are:

- Severe class imbalance
- Visual similarity between lesion types
- Limited number of samples for rare classes
- Risk of data leakage if patient-level splitting is not used
- Need for strong evaluation metrics beyond normal accuracy

## Project Workflow

The complete workflow includes:

1. Loading the HAM10000 dataset using `kagglehub`
2. Reading metadata and mapping image IDs to image paths
3. Performing exploratory data analysis
4. Visualizing class distribution and patient demographics
5. Splitting the dataset into train, validation, and test sets
6. Applying data augmentation for training images
7. Handling class imbalance using weighted sampling and class-weighted loss
8. Training baseline and deep learning models
9. Evaluating models using balanced accuracy, macro F1, and AUC-ROC
10. Comparing final model performance using visualizations

## Exploratory Data Analysis

The notebook includes several visualizations to understand the dataset:

- Class distribution chart
- Class share donut chart
- Patient age distribution by lesion type
- Sex distribution across lesion classes
- Lesion localization distribution
- Sample images and mean image per class
- RGB channel histograms
- Brightness and contrast distribution
- Augmentation effect on sample images

The dataset is highly imbalanced. The `nv` class represents the majority of the images, while classes like `df` and `vasc` have very few samples. This makes balanced accuracy and macro F1 more meaningful than simple accuracy.

## Data Splitting Strategy

The dataset is split by `lesion_id` instead of random image-level splitting.

This is important because the same lesion or patient may have multiple images. If similar images appear in both training and testing sets, the model can show artificially high performance. Patient-level or lesion-level splitting helps reduce data leakage.

| Split | Purpose |
|---|---|
| Train | Used for model learning |
| Validation | Used for tuning and early stopping |
| Test | Used for final evaluation |

## Preprocessing and Augmentation

Training images are resized and augmented to improve generalization.

Common augmentations used:

- Random horizontal flip
- Random vertical flip
- Random rotation
- Color jitter
- Image normalization

Validation and test images are resized and normalized without random augmentation.

## Imbalance Handling

The project uses two main techniques to handle class imbalance:

| Technique | Purpose |
|---|---|
| Class-weighted loss | Penalizes errors on minority classes more strongly |
| WeightedRandomSampler | Oversamples minority classes during training batches |

This helps the model learn rare classes instead of only predicting the majority class.

## Models Implemented

The notebook compares six model approaches.

| Model | Type | Description |
|---|---|---|
| Most Frequent Class | Statistical baseline | Always predicts the majority class |
| Stratified Random | Statistical baseline | Predicts based on class distribution |
| Simple 3-Layer CNN | Neural baseline | Basic convolutional model trained from scratch |
| Custom CNN with DSConv | Proposed model | Uses depthwise separable convolutions |
| ResNet-50 | Transfer learning | Fine-tuned ImageNet-pretrained ResNet model |
| EfficientNet-B3 | Transfer learning | Fine-tuned ImageNet-pretrained EfficientNet model |

## Custom CNN Architecture

The custom model uses depthwise separable convolutions inspired by MobileNet-style architectures.

Key design points:

- Convolution stem layer
- Multiple depthwise separable convolution blocks
- Batch normalization
- SiLU activation
- Dropout for regularization
- Global average pooling
- Fully connected classification head

This model is designed to be lighter than large transfer learning models while still performing better than a basic CNN.

## Training Configuration

The training pipeline includes:

| Component | Method |
|---|---|
| Loss Function | CrossEntropyLoss with class weights |
| Optimizer | AdamW |
| Learning Rate Scheduler | Cosine Annealing with Warm Restarts |
| Early Stopping | Based on validation balanced accuracy |
| Checkpointing | Best validation model is restored |
| Device | CUDA GPU if available, otherwise CPU |

## Evaluation Metrics

The project uses metrics suitable for imbalanced classification:

| Metric | Reason |
|---|---|
| Balanced Accuracy | Treats all classes more equally |
| Macro F1 Score | Measures average F1 across all classes |
| AUC-ROC | Evaluates probability-based classification performance |
| Per-Class F1 | Shows how each class performs individually |
| Confusion Matrix | Shows which classes are confused with each other |
| Melanoma Recall | Important because missing melanoma is clinically serious |

## Expected Results

The expected model ranking is:

| Model | Expected Performance |
|---|---|
| Most Frequent Class | Lowest performance |
| Stratified Random | Low baseline |
| Simple CNN | Better than statistical baselines |
| Custom CNN | Better than Simple CNN |
| ResNet-50 | Strong transfer learning model |
| EfficientNet-B3 | Best overall model |

EfficientNet-B3 is expected to achieve the best balanced accuracy and macro F1 score because it uses strong pretrained image features and fine-tuning.

## Key Findings

The main findings from the project are:

- Transfer learning performs better than training from scratch.
- EfficientNet-B3 gives the best overall classification performance.
- Class imbalance remains the biggest challenge.
- Minority classes such as `df` and `vasc` are harder to classify.
- Weighted sampling and class-weighted loss improve minority-class learning.
- Patient-level splitting is important to avoid data leakage.
- Melanoma recall is a key metric for medical relevance.

## Generated Outputs

The notebook saves or displays the following outputs:

| Output | Description |
|---|---|
| `final_results.csv` | Final model comparison table |
| `viz1_class_distribution.png` | Class distribution visualization |
| `viz8_f1_heatmap.png` | Per-class F1 heatmap |
| `viz9_precision_recall.png` | Precision-recall curves |
| `viz10_confidence_distribution.png` | Confidence distribution plot |
| `viz11_confused_pairs.png` | Most confused class pairs |
| `viz13_final_comparison.png` | Final model comparison chart |
| `viz14_radar_chart.png` | Radar chart of model strengths |
| `simple_cnn.pth` | Saved Simple CNN model weights |
| `custom_cnn.pth` | Saved Custom CNN model weights |
| `resnet50.pth` | Saved ResNet-50 model weights |
| `efficientnet_b3.pth` | Saved EfficientNet-B3 model weights |

## Repository Structure

A suggested repository structure is:

```text
skin-lesion-classification/
│
├── skin_lesion_Classification_9.ipynb
├── README.md
├── final_results.csv
├── requirements.txt
│
├── outputs/
│   ├── viz1_class_distribution.png
│   ├── viz8_f1_heatmap.png
│   ├── viz9_precision_recall.png
│   ├── viz10_confidence_distribution.png
│   ├── viz11_confused_pairs.png
│   ├── viz13_final_comparison.png
│   └── viz14_radar_chart.png
│
└── models/
    ├── simple_cnn.pth
    ├── custom_cnn.pth
    ├── resnet50.pth
    └── efficientnet_b3.pth
```

## Requirements

Install the required libraries before running the notebook.

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn torch torchvision timm kagglehub
```

If you are using Google Colab, most packages are already available. You may only need:

```bash
pip install -q timm kagglehub
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Open the notebook:

```bash
jupyter notebook skin_lesion_Classification_9.ipynb
```

3. Run all cells from top to bottom.

4. The dataset will be downloaded automatically using `kagglehub`.

5. After training, check the generated evaluation graphs and `final_results.csv`.

## Kaggle Dataset Access

The notebook uses:

```python
DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"
```

If the public Kaggle download does not work, configure your Kaggle API token.

For local use:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

For Google Colab, add the Kaggle API token through Colab Secrets or upload `kaggle.json`.

## GitHub Notebook Preview Issue

If GitHub shows an "Invalid Notebook" error, it is usually caused by broken widget metadata in the `.ipynb` file.

A simple fix is to clear notebook outputs and remove widget metadata before uploading:

```python
import json

input_file = "skin_lesion_Classification_9.ipynb"
output_file = "skin_lesion_Classification_9_fixed.ipynb"

with open(input_file, "r", encoding="utf-8") as f:
    notebook = json.load(f)

if "metadata" in notebook and "widgets" in notebook["metadata"]:
    del notebook["metadata"]["widgets"]

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Fixed notebook saved as:", output_file)
```

Upload the fixed notebook to GitHub.

## Future Improvements

Possible improvements include:

- Use focal loss for stronger imbalance handling
- Add Grad-CAM visualizations for explainability
- Try test-time augmentation
- Fine-tune the full EfficientNet model with a very low learning rate
- Build an ensemble of ResNet-50 and EfficientNet-B3
- Add cross-validation for more reliable evaluation
- Use segmentation-based preprocessing to focus only on lesion regions
- Deploy the best model as a simple web application

## Limitations

This project is for academic and research purposes only. It should not be used as a medical diagnosis system. Real clinical deployment would require expert validation, larger datasets, explainability checks, fairness testing, and regulatory approval.

## Conclusion

This project demonstrates an end-to-end deep learning workflow for skin lesion classification using the HAM10000 dataset. It compares simple baselines, custom CNN models, and transfer learning models. The results show that transfer learning, especially EfficientNet-B3, is the most effective approach for this imbalanced medical image classification task.
