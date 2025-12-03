# Amazon ML Challenge - Product Price Prediction

This project aims to predict the price of Amazon products using a multi-modal approach that leverages product images, catalog descriptions, and other tabular features. The solution involves several stages: data cleaning, image downloading, feature extraction via embeddings, and final price prediction using a deep learning model.

## 🚀 Project Workflow

The end-to-end pipeline is executed through a series of Jupyter notebooks:

1.  **Data Cleaning & Preparation (`cleaned_data.ipynb`)**: Raw catalog data is cleaned, structured, and merged. Key features like product name, value, and unit are extracted.
2.  **Image Downloading (`download image.ipynb`)**: Product images are downloaded from the URLs provided in the dataset.
3.  **Embedding Extraction (in `cleaned_data.ipynb`)**: Multi-modal embeddings are generated from the cleaned text and downloaded images using pre-trained transformer models.
4.  **Price Prediction (`third_training.ipynb`)**: A pre-trained ResNet-based regression model uses the generated embeddings and other features to predict the final product prices.

---

## 📂 Directory Structure

```
Amazon-ML-Challenge-2025/
├── dataset/
│   ├── train.csv
│   └── test.csv
├── images/
│   └── train/
│       └── (*.jpg)
├── test_images/
│   └── test_images/
│       └── (*.jpg)
├── test_embeddings/
│   └── batch_*/
│       ├── image_embeddings.pt
│       └── text_embeddings.pt
├── fast_ensemble_models/
│   ├── resnet_model.pth
│   ├── scaler.pkl
│   └── unit_encoder.pkl
├── cleaned_data.ipynb
├── download image.ipynb
├── third_training.ipynb
└── README.md
```

---

## ⚙️ Setup and Requirements

First, ensure you have the required Python libraries installed.

```bash
pip install pandas torch torchvision transformers scikit-learn pillow tqdm requests
```

---

## 🏃‍♂️ How to Run the Pipeline

Follow these steps in order to reproduce the predictions.

### Step 1: Clean and Prepare the Data

Run the cells in **`cleaned_data.ipynb`** sequentially. This notebook performs several key tasks:
*   Reads the raw data from `dataset/test.csv`.
*   Cleans the `catalog_content` field by normalizing text, removing irrelevant prefixes (e.g., "Bullet Point:"), and structuring the content.
*   Extracts the `product_name`, `value`, and `unit` from the cleaned text.
*   Standardizes measurement units (e.g., "oz" and "ounces" both become "ounce").
*   Merges all cleaned and extracted features into a single file: `test/merged_test_with_image.csv`. This file is crucial for the subsequent steps.

### Step 2: Download Product Images

Run the script in **`download image.ipynb`**.
*   This script reads the `dataset/test.csv` file.
*   It iterates through each product, downloads the corresponding image from the `image_link` URL, and saves it.
*   Images are saved in the `test_images/test_images/` directory with the filename `{sample_id}.jpg`.

*Note: The notebook may need to be adjusted to point to `dataset/test.csv` and save to the correct test image directory.*

### Step 3: Generate Multi-Modal Embeddings

The final cells in **`cleaned_data.ipynb`** are responsible for this step. These cells set up a pipeline to generate embeddings but may have been interrupted.

*   **Image Embeddings**: Uses the `facebook/dinov2-base` model to generate embeddings from the product images located in `test_images/test_images/`.
*   **Text Embeddings**: Uses the `sentence-transformers/all-MiniLM-L12-v2` model to generate embeddings from the `catalog_content_clean` column in `test/merged_test_with_image.csv`.
*   The script processes the data in batches and saves the resulting image and text embeddings into the `./test_embeddings/` directory.

*Note: This is a computationally intensive step. Ensure you have a CUDA-compatible GPU for faster processing. The script is designed to be resumable.*

### Step 4: Predict Prices

Finally, run the **`third_training.ipynb`** notebook to generate the final predictions.

*   It loads the image and text embeddings generated in the previous step.
*   It loads the other features (`value`, `unit`) from `test/merged_test_with_image.csv`.
*   The features are scaled and one-hot encoded using the pre-fitted `scaler.pkl` and `unit_encoder.pkl` from the `fast_ensemble_models/` directory.
*   The pre-trained `ResNetRegressor` model (`resnet_model.pth`) is loaded.
*   The model predicts prices for the test set.
*   The final predictions are saved to **`resnet_predictions.csv`** with `sample_id` and `price` columns.

---

## 🤖 Models Used

*   **Image Feature Extraction**: `facebook/dinov2-base` - A powerful vision transformer for generating high-quality image representations.
*   **Text Feature Extraction**: `sentence-transformers/all-MiniLM-L12-v2` - An efficient sentence-transformer model for creating semantic text embeddings.
*   **Price Regression**: A custom `ResNetRegressor` built with PyTorch, which uses residual blocks to effectively learn from the combined image, text, and tabular features.



