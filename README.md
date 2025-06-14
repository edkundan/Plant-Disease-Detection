# Plant Disease Detection 

## Project Overview

This project focuses on developing a robust deep learning model for the automated detection and classification of various plant diseases from leaf images. Early and accurate identification of plant diseases is crucial for effective agricultural management, preventing crop loss, and ensuring food security. This system aims to provide a fast, scalable, and accessible solution compared to traditional manual inspection methods.

We leverage **Transfer Learning** with a state-of-the-art Convolutional Neural Network (CNN) to achieve high accuracy in classifying different plant diseases.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Methodology](#methodology)
-   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
-   [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
-   [Model Architecture](#model-architecture)
-   [Training & Optimization](#training--optimization)
-   [Results & Evaluation](#results--evaluation)
-   [Data Visualizations](#data-visualizations)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [Dataset Setup (Crucial!)](#dataset-setup-crucial)
    -   [How to Run the Notebook](#how-to-run-the-notebook)
-   [Project Structure](#project-structure)
-   [Code Quality](#code-quality)
-   [Future Enhancements](#future-enhancements)
-   [License](#license)
-   [Contact](#contact)

---

## Dataset

This project utilizes the **PlantVillage Dataset**, a widely recognized and publicly available collection of plant leaf images.

* **Source:** [Kaggle - Plant Disease Dataset by Emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Content:** The dataset comprises images of various plant species (e.g., Tomato, Potato, Pepper) affected by different diseases, as well as healthy leaves.
* **Total Images:** Approximately **20639** (sum of counts from all classes)
* **Total Classes:** **15** unique plant-disease categories.
* **Characteristics:** The dataset exhibits a degree of class imbalance, which is addressed during the data preprocessing and model training phases.

## Methodology

Our approach to plant disease detection follows a standard deep learning pipeline:

1.  **Data Collection & Initial Structuring:** Gathering image file paths and their corresponding labels into a structured Pandas DataFrame.
2.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of classes and inspecting sample images to understand the dataset's characteristics.
3.  **Data Preprocessing:**
    * Splitting the dataset into training, validation, and testing subsets with stratification to maintain class distribution.
    * Encoding categorical labels into numerical formats suitable for machine learning.
    * Implementing extensive **Data Augmentation** techniques to enhance dataset diversity and prevent overfitting.
4.  **Model Building:** Constructing a Convolutional Neural Network (CNN) using **Transfer Learning** from a powerful pre-trained model (InceptionV3).
5.  **Model Training:** Training the model with optimized parameters, including class weighting to handle imbalance, and advanced callbacks for efficient learning.
6.  **Model Evaluation:** Rigorously assessing the model's performance using various metrics (accuracy, loss, precision, recall, F1-score) and visualizations (confusion matrix, training history plots).

## Exploratory Data Analysis (EDA)

Initial analysis of the PlantVillage dataset revealed the distribution of different plant disease categories.

* **Class Distribution:** We observed varying numbers of images per class, indicating some class imbalance. This was visualized using:
    * **Pie Chart:** Showing the proportional representation of each disease category.
    * **Horizontal Bar Chart:** Clearly displaying the absolute counts for each category, ordered for easy comparison.
    * **Vertical Bar Chart:** Providing another perspective on the frequency of each disease.
* **Sample Images:** Visual inspection of sample images from each class helped us understand the visual characteristics of different diseases and the overall quality of the dataset. This included observing variations in lighting, background, and disease severity.

## Data Preprocessing & Augmentation

Effective data preparation is critical for training robust deep learning models.

* **Dataset Splitting:** The dataset was divided into:
    * **Training Set:** 80% of data (for model learning).
    * **Validation Set:** 10% of data (for hyperparameter tuning and early stopping).
    * **Test Set:** 10% of data (for final, unbiased evaluation).
    * **Stratification:** Crucially, `stratify` was used during splitting to ensure that the proportion of each disease class is maintained across all three subsets, preserving data integrity and consistency.
* **Label Encoding:** Textual disease labels were converted into numerical format using `LabelEncoder` for compatibility with the neural network.
* **Image Data Augmentation:** To combat overfitting and improve the model's generalization capabilities, extensive real-time image augmentation was applied to the training data using `ImageDataGenerator`. Techniques included:
    * Rescaling pixel values (normalization).
    * Random rotations, width/height shifts, shear, and zoom.
    * Horizontal and vertical flipping.
    * Brightness and channel shift adjustments.
    * This process effectively creates new, varied training examples from existing ones, acting as a powerful form of feature engineering.

## Model Architecture

The core of our plant disease classifier is built upon **Transfer Learning**, leveraging the power of pre-trained Convolutional Neural Networks (CNNs).

* **Base Model:** We utilized **InceptionV3**, a deep CNN architecture pre-trained on the vast ImageNet dataset. This model serves as a highly effective feature extractor, learning complex visual patterns from images.
* **Custom Classification Head:** The top layers of InceptionV3 were removed, and custom layers were added to adapt the model for our specific 15-class plant disease classification task:
    * `GlobalAveragePooling2D`: Reduces the dimensionality of the feature maps, making the model more robust to variations in object position.
    * `Dense` layer (1024 units, ReLU activation): A fully connected layer to learn higher-level combinations of features.
    * `Dense` output layer (15 units, Softmax activation): The final layer, outputting probabilities for each of the 15 disease classes.
* **Optimizer:** Adam optimizer was used for efficient gradient descent during training.
* **Loss Function:** `Categorical Crossentropy` was chosen as the loss function, suitable for multi-class classification problems.

## Training & Optimization

The model was trained with careful consideration for efficiency and performance.

* **Training Parameters:**
    * **Epochs:** 10 (with Early Stopping to prevent overfitting).
    * **Batch Size:** 32.
    * **Hardware Acceleration:** Training was performed on a GPU (e.g., NVIDIA Tesla P100 in Google Colab) to significantly accelerate the process.
* **Class Weighting:** To address the class imbalance observed in the dataset, class weights were computed and applied during training. This ensures that the model pays more attention to under-represented classes, preventing them from being ignored.
* **Callbacks:** Several Keras callbacks were implemented to optimize the training process:
    * `EarlyStopping`: Monitors validation loss and stops training if performance doesn't improve for a set number of epochs, restoring the best weights. This prevents overfitting and saves computational resources.
    * `ReduceLROnPlateau`: Reduces the learning rate when validation loss plateaus, allowing the model to fine-tune its weights more precisely.
    * `ModelCheckpoint`: Automatically saves the model's weights (or the entire model) at the end of each epoch if it achieves the best validation loss so far.

## Results & Evaluation

The trained model's performance was rigorously evaluated on the unseen test set.

* **Overall Performance:**
    * **Test Accuracy:** **99.08%**
    * **Test Loss:** **0.0288**
* **Training History:** Plots of training and validation loss/accuracy over epochs demonstrate the model's learning progression and convergence. The plots show a consistent decrease in both training and validation loss, and an increase in accuracy, indicating successful learning and minimal overfitting.
* **Confusion Matrix:** A heatmap visualizing the model's predictions against the true labels. This provides a detailed breakdown of correct classifications and specific misclassifications between classes. The confusion matrix highlights high accuracy across most classes, with minor confusions primarily occurring between visually similar disease types.
* **Classification Report:** Provides per-class metrics including Precision, Recall, and F1-score, offering a granular view of the model's performance for each disease category. The classification report confirms high precision and recall for most classes, indicating the model's strong ability to correctly identify diseases and minimize false positives/negatives.

---

## Data Visualizations

Our project incorporates comprehensive data visualizations to effectively communicate insights from the dataset, model training, and final evaluation. These visuals are designed to provide a clear narrative and deepen understanding of the plant disease detection system.

**Objectives of our Visualizations:**

* **Understand Dataset Characteristics:** To visualize the distribution of plant disease classes, helping to identify potential data imbalances.
* **Monitor Model Learning Progress:** To track the model's performance during training (accuracy and loss) and detect signs of overfitting or underfitting.
* **Assess Model Performance in Detail:** To evaluate the model's classification accuracy comprehensively, identifying strengths and areas of confusion between different disease categories.
* **Provide Granular Performance Metrics:** To present per-class precision, recall, and F1-score, offering a fine-grained view of the model's reliability for each specific disease.

### How to Interact with the Visualizations

All core visualizations are generated within the main Jupyter Notebook: `notebooks/Plant_Disease_Detection.ipynb`.

**To experience the full interactivity of the plots (especially if `plotly.express` was used for some visualizations):**

1.  **Open the Notebook in Google Colab (Recommended) or a Jupyter environment:**
    * Navigate to `notebooks/Plant_Disease_Detection.ipynb`.
    * Upload it to Google Colab, or open it in your local Jupyter Lab/Notebook.
2.  **Run All Cells Sequentially:** Ensure all cells from Section 1 through Section 7 (the visualization section) are executed.
3.  **Explore Interactive Plots:**
    * For plots generated using `plotly.express` (e.g., Dataset Distribution, Training/Validation History, or Confusion Matrix if enabled), you can:
        * **Hover** over data points (bars, lines, heatmap cells) to see exact values and details in tooltips.
        * **Zoom In/Out:** Use your mouse wheel or click-and-drag to select an area for zooming. Double-click to reset zoom.
        * **Pan:** Click and drag the plot to move around the zoomed area.
        * **Toggle Data Series:** Click on legend items (e.g., "Training Accuracy", "Validation Accuracy") to hide or show specific lines on the graph, allowing focused analysis.
    * Static plots (generated with `matplotlib` or `seaborn`) provide clear, high-resolution insights directly.

These interactive elements enhance data exploration and allow for a more dynamic review of the project's findings.

---

## Getting Started

Follow these instructions to set up the environment and run the project.

### Prerequisites

* **Python:** Version 3.8 or higher.
* **Google Colab:** Recommended for access to free GPUs, which are essential for training deep learning models efficiently.
* **Jupyter Notebook/Lab:** If running locally (GPU recommended).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/edkundan/Plant-Disease-Detection
    cd Plant-Disease-Detection
    ```
2.  **Install required Python packages:**
    It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) for dependency management.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

### Dataset Setup (Crucial!)

The PlantVillage dataset needs to be downloaded from Kaggle. The notebook contains integrated commands for this.

1.  **Generate Kaggle API Token:** Go to [Kaggle.com](https://www.kaggle.com/) -> Your Profile -> Account -> "Create New API Token". This downloads `kaggle.json`.
2.  **Upload `kaggle.json` to Colab:** Run `from google.colab import files; files.upload()` in a Colab cell.
3.  **Configure Kaggle Credentials:** Run `!mkdir -p ~/.kaggle; !mv kaggle.json ~/.kaggle/; !chmod 600 ~/.kaggle/kaggle.json` in a Colab cell.
4.  **Download the Dataset:** Run `!kaggle datasets download -d emmarex/plantdisease` in a Colab cell.
5.  **Unzip the Dataset:** Run `!unzip -q plantdisease.zip -d /content/plant_disease_dataset/` in a Colab cell.
    * This will create the dataset structure at `/content/plant_disease_dataset/PlantVillage`. The notebook is configured to use this path.

### How to Run the Notebook

1.  **Open the Jupyter Notebook:**
    * If using Google Colab, upload `notebooks/Plant_Disease_Detection.ipynb` to your Colab environment.
    * If running locally, navigate to the project directory in your terminal and run `jupyter notebook`. Open `notebooks/Plant_Disease_Detection.ipynb`.
2.  **Execute Cells Sequentially:** Run each code cell in the notebook from top to bottom.
    * **Section 1:** Imports and initial setup.
    * **Section 2:** Dataset loading (ensure dataset is set up as per instructions above).
    * **Section 3:** Exploratory Data Analysis & Visualizations.
    * **Section 4:** Data Preprocessing for Deep Learning.
    * **Section 5:** Model Definition & Training (or **Loading** if you have a saved model). This section will take significant time for training, but is fast if loading a pre-trained model. Ensure GPU runtime is enabled in Colab: `Runtime > Change runtime type > GPU`.
    * **Section 6:** Model Evaluation & Performance Visualization.
    * **Section 7:** Comprehensive Data Visualizations and Storytelling.

## Project Structure

Here is the complete and updated README.md file content, incorporating all the sections and refinements we've discussed, including the corrected project structure formatting and the detailed explanation for data visualizations and their interactivity.

Please copy this entire block of text and save it as README.md in the root directory of your GitHub repository. Remember to replace placeholder values like [Your Full Name], [Your Email Address], [Your LinkedIn Profile URL], and YOUR_GITHUB_USERNAME with your actual information.

Markdown

# Plant Disease Detection using Deep Learning

## Project Overview

This project focuses on developing a robust deep learning model for the automated detection and classification of various plant diseases from leaf images. Early and accurate identification of plant diseases is crucial for effective agricultural management, preventing crop loss, and ensuring food security. This system aims to provide a fast, scalable, and accessible solution compared to traditional manual inspection methods.

We leverage **Transfer Learning** with a state-of-the-art Convolutional Neural Network (CNN) to achieve high accuracy in classifying different plant diseases.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Methodology](#methodology)
-   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
-   [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
-   [Model Architecture](#model-architecture)
-   [Training & Optimization](#training--optimization)
-   [Results & Evaluation](#results--evaluation)
-   [Data Visualizations](#data-visualizations)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
    -   [Dataset Setup (Crucial!)](#dataset-setup-crucial)
    -   [How to Run the Notebook](#how-to-run-the-notebook)
-   [Project Structure](#project-structure)
-   [Code Quality](#code-quality)
-   [Future Enhancements](#future-enhancements)
-   [License](#license)
-   [Contact](#contact)

---

## Dataset

This project utilizes the **PlantVillage Dataset**, a widely recognized and publicly available collection of plant leaf images.

* **Source:** [Kaggle - Plant Disease Dataset by Emmarex](https://www.kaggle.com/datasets/emmarex/plantdisease)
* **Content:** The dataset comprises images of various plant species (e.g., Tomato, Potato, Pepper) affected by different diseases, as well as healthy leaves.
* **Total Images:** Approximately **20639** (sum of counts from all classes)
* **Total Classes:** **15** unique plant-disease categories.
* **Characteristics:** The dataset exhibits a degree of class imbalance, which is addressed during the data preprocessing and model training phases.

## Methodology

Our approach to plant disease detection follows a standard deep learning pipeline:

1.  **Data Collection & Initial Structuring:** Gathering image file paths and their corresponding labels into a structured Pandas DataFrame.
2.  **Exploratory Data Analysis (EDA):** Visualizing the distribution of classes and inspecting sample images to understand the dataset's characteristics.
3.  **Data Preprocessing:**
    * Splitting the dataset into training, validation, and testing subsets with stratification to maintain class distribution.
    * Encoding categorical labels into numerical formats suitable for machine learning.
    * Implementing extensive **Data Augmentation** techniques to enhance dataset diversity and prevent overfitting.
4.  **Model Building:** Constructing a Convolutional Neural Network (CNN) using **Transfer Learning** from a powerful pre-trained model (InceptionV3).
5.  **Model Training:** Training the model with optimized parameters, including class weighting to handle imbalance, and advanced callbacks for efficient learning.
6.  **Model Evaluation:** Rigorously assessing the model's performance using various metrics (accuracy, loss, precision, recall, F1-score) and visualizations (confusion matrix, training history plots).

## Exploratory Data Analysis (EDA)

Initial analysis of the PlantVillage dataset revealed the distribution of different plant disease categories.

* **Class Distribution:** We observed varying numbers of images per class, indicating some class imbalance. This was visualized using:
    * **Pie Chart:** Showing the proportional representation of each disease category.
    * **Horizontal Bar Chart:** Clearly displaying the absolute counts for each category, ordered for easy comparison.
    * **Vertical Bar Chart:** Providing another perspective on the frequency of each disease.
* **Sample Images:** Visual inspection of sample images from each class helped us understand the visual characteristics of different diseases and the overall quality of the dataset. This included observing variations in lighting, background, and disease severity.

## Data Preprocessing & Augmentation

Effective data preparation is critical for training robust deep learning models.

* **Dataset Splitting:** The dataset was divided into:
    * **Training Set:** 80% of data (for model learning).
    * **Validation Set:** 10% of data (for hyperparameter tuning and early stopping).
    * **Test Set:** 10% of data (for final, unbiased evaluation).
    * **Stratification:** Crucially, `stratify` was used during splitting to ensure that the proportion of each disease class is maintained across all three subsets, preserving data integrity and consistency.
* **Label Encoding:** Textual disease labels were converted into numerical format using `LabelEncoder` for compatibility with the neural network.
* **Image Data Augmentation:** To combat overfitting and improve the model's generalization capabilities, extensive real-time image augmentation was applied to the training data using `ImageDataGenerator`. Techniques included:
    * Rescaling pixel values (normalization).
    * Random rotations, width/height shifts, shear, and zoom.
    * Horizontal and vertical flipping.
    * Brightness and channel shift adjustments.
    * This process effectively creates new, varied training examples from existing ones, acting as a powerful form of feature engineering.

## Model Architecture

The core of our plant disease classifier is built upon **Transfer Learning**, leveraging the power of pre-trained Convolutional Neural Networks (CNNs).

* **Base Model:** We utilized **InceptionV3**, a deep CNN architecture pre-trained on the vast ImageNet dataset. This model serves as a highly effective feature extractor, learning complex visual patterns from images.
* **Custom Classification Head:** The top layers of InceptionV3 were removed, and custom layers were added to adapt the model for our specific 15-class plant disease classification task:
    * `GlobalAveragePooling2D`: Reduces the dimensionality of the feature maps, making the model more robust to variations in object position.
    * `Dense` layer (1024 units, ReLU activation): A fully connected layer to learn higher-level combinations of features.
    * `Dense` output layer (15 units, Softmax activation): The final layer, outputting probabilities for each of the 15 disease classes.
* **Optimizer:** Adam optimizer was used for efficient gradient descent during training.
* **Loss Function:** `Categorical Crossentropy` was chosen as the loss function, suitable for multi-class classification problems.

## Training & Optimization

The model was trained with careful consideration for efficiency and performance.

* **Training Parameters:**
    * **Epochs:** 10 (with Early Stopping to prevent overfitting).
    * **Batch Size:** 32.
    * **Hardware Acceleration:** Training was performed on a GPU (e.g., NVIDIA Tesla P100 in Google Colab) to significantly accelerate the process.
* **Class Weighting:** To address the class imbalance observed in the dataset, class weights were computed and applied during training. This ensures that the model pays more attention to under-represented classes, preventing them from being ignored.
* **Callbacks:** Several Keras callbacks were implemented to optimize the training process:
    * `EarlyStopping`: Monitors validation loss and stops training if performance doesn't improve for a set number of epochs, restoring the best weights. This prevents overfitting and saves computational resources.
    * `ReduceLROnPlateau`: Reduces the learning rate when validation loss plateaus, allowing the model to fine-tune its weights more precisely.
    * `ModelCheckpoint`: Automatically saves the model's weights (or the entire model) at the end of each epoch if it achieves the best validation loss so far.

## Results & Evaluation

The trained model's performance was rigorously evaluated on the unseen test set.

* **Overall Performance:**
    * **Test Accuracy:** **99.08%**
    * **Test Loss:** **0.0288**
* **Training History:** Plots of training and validation loss/accuracy over epochs demonstrate the model's learning progression and convergence. The plots show a consistent decrease in both training and validation loss, and an increase in accuracy, indicating successful learning and minimal overfitting.
* **Confusion Matrix:** A heatmap visualizing the model's predictions against the true labels. This provides a detailed breakdown of correct classifications and specific misclassifications between classes. The confusion matrix highlights high accuracy across most classes, with minor confusions primarily occurring between visually similar disease types.
* **Classification Report:** Provides per-class metrics including Precision, Recall, and F1-score, offering a granular view of the model's performance for each disease category. The classification report confirms high precision and recall for most classes, indicating the model's strong ability to correctly identify diseases and minimize false positives/negatives.

---

## Data Visualizations

Our project incorporates comprehensive data visualizations to effectively communicate insights from the dataset, model training, and final evaluation. These visuals are designed to provide a clear narrative and deepen understanding of the plant disease detection system.

**Objectives of our Visualizations:**

* **Understand Dataset Characteristics:** To visualize the distribution of plant disease classes, helping to identify potential data imbalances.
* **Monitor Model Learning Progress:** To track the model's performance during training (accuracy and loss) and detect signs of overfitting or underfitting.
* **Assess Model Performance in Detail:** To evaluate the model's classification accuracy comprehensively, identifying strengths and areas of confusion between different disease categories.
* **Provide Granular Performance Metrics:** To present per-class precision, recall, and F1-score, offering a fine-grained view of the model's reliability for each specific disease.

### How to Interact with the Visualizations

All core visualizations are generated within the main Jupyter Notebook: `notebooks/Plant_Disease_Detection.ipynb`.

**To experience the full interactivity of the plots (especially if `plotly.express` was used for some visualizations):**

1.  **Open the Notebook in Google Colab (Recommended) or a Jupyter environment:**
    * Navigate to `notebooks/Plant_Disease_Detection.ipynb`.
    * Upload it to Google Colab, or open it in your local Jupyter Lab/Notebook.
2.  **Run All Cells Sequentially:** Ensure all cells from Section 1 through Section 7 (the visualization section) are executed.
3.  **Explore Interactive Plots:**
    * For plots generated using `plotly.express` (e.g., Dataset Distribution, Training/Validation History, or Confusion Matrix if enabled), you can:
        * **Hover** over data points (bars, lines, heatmap cells) to see exact values and details in tooltips.
        * **Zoom In/Out:** Use your mouse wheel or click-and-drag to select an area for zooming. Double-click to reset zoom.
        * **Pan:** Click and drag the plot to move around the zoomed area.
        * **Toggle Data Series:** Click on legend items (e.g., "Training Accuracy", "Validation Accuracy") to hide or show specific lines on the graph, allowing focused analysis.
    * Static plots (generated with `matplotlib` or `seaborn`) provide clear, high-resolution insights directly.

These interactive elements enhance data exploration and allow for a more dynamic review of the project's findings.

---

## Getting Started

Follow these instructions to set up the environment and run the project.

### Prerequisites

* **Python:** Version 3.8 or higher.
* **Google Colab:** Recommended for access to free GPUs, which are essential for training deep learning models efficiently.
* **Jupyter Notebook/Lab:** If running locally (GPU recommended).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/Plant-Disease-Detection.git](https://github.com/YOUR_GITHUB_USERNAME/Plant-Disease-Detection.git)
    cd Plant-Disease-Detection
    ```
2.  **Install required Python packages:**
    It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) for dependency management.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

### Dataset Setup (Crucial!)

The PlantVillage dataset needs to be downloaded from Kaggle. The notebook contains integrated commands for this.

1.  **Generate Kaggle API Token:** Go to [Kaggle.com](https://www.kaggle.com/) -> Your Profile -> Account -> "Create New API Token". This downloads `kaggle.json`.
2.  **Upload `kaggle.json` to Colab:** Run `from google.colab import files; files.upload()` in a Colab cell.
3.  **Configure Kaggle Credentials:** Run `!mkdir -p ~/.kaggle; !mv kaggle.json ~/.kaggle/; !chmod 600 ~/.kaggle/kaggle.json` in a Colab cell.
4.  **Download the Dataset:** Run `!kaggle datasets download -d emmarex/plantdisease` in a Colab cell.
5.  **Unzip the Dataset:** Run `!unzip -q plantdisease.zip -d /content/plant_disease_dataset/` in a Colab cell.
    * This will create the dataset structure at `/content/plant_disease_dataset/PlantVillage`. The notebook is configured to use this path.

### How to Run the Notebook

1.  **Open the Jupyter Notebook:**
    * If using Google Colab, upload `notebooks/Plant_Disease_Detection.ipynb` to your Colab environment.
    * If running locally, navigate to the project directory in your terminal and run `jupyter notebook`. Open `notebooks/Plant_Disease_Detection.ipynb`.
2.  **Execute Cells Sequentially:** Run each code cell in the notebook from top to bottom.
    * **Section 1:** Imports and initial setup.
    * **Section 2:** Dataset loading (ensure dataset is set up as per instructions above).
    * **Section 3:** Exploratory Data Analysis & Visualizations.
    * **Section 4:** Data Preprocessing for Deep Learning.
    * **Section 5:** Model Definition & Training (or **Loading** if you have a saved model). This section will take significant time for training, but is fast if loading a pre-trained model. Ensure GPU runtime is enabled in Colab: `Runtime > Change runtime type > GPU`.
    * **Section 6:** Model Evaluation & Performance Visualization.
    * **Section 7:** Comprehensive Data Visualizations and Storytelling.

## Project Structure

Aapke project ki file aur folder structure is prakaar hai:

.
├── Plant_Disease_Detection.ipynb  # Main project Jupyter Notebook
├── models/
│   └── plant_disease_classifier.keras # Saved trained Keras model
├── data/
│   └── PlantVillage/                  # Placeholder for the dataset (actual data is downloaded by notebook instructions)
│       └── ... (subfolders for each plant/disease class)
├── requirements.txt                   # List of Python dependencies
└── README.md                          # Project documentation (this file)

## Code Quality

The code in this repository adheres to the following quality standards:

* **Well-Commented:** Extensive comments are provided throughout the notebook, explaining the purpose of each section, complex logic, and key design choices. Comments also explicitly link code sections to the project's rubric requirements.
* **Modular:** The project utilizes functions for reusable components (e.g., `make_pandas_dataset`, `show_image_grid`, `create_inceptionv3_model`, `plot_training_history`), enhancing readability and maintainability.
* **Readability:** Code follows general PEP 8 guidelines for naming conventions (`snake_case` for variables/functions, `PascalCase` for classes), consistent indentation (4 spaces), and clear logical separation.
* **Error Handling:** Basic `try-except` blocks are used where file operations might fail (e.g., image loading in `show_image_grid`, model loading).

## Future Enhancements

* **Explore Advanced Architectures:** Experiment with newer or more efficient CNN models (e.g., EfficientNet, Vision Transformers) for potentially higher accuracy or faster inference.
* **Real-time Inference:** Develop a lightweight version of the model for deployment on mobile devices or edge computing platforms for real-time disease detection in the field.
* **Dataset Expansion:** Incorporate more diverse images (different lighting conditions, backgrounds, disease stages, multiple diseases on one leaf) to improve model robustness.
* **Explainable AI (XAI):** Implement techniques like Grad-CAM or LIME to visualize which parts of the image the model focuses on for its predictions, increasing trust and interpretability.
* **Web Application/API:** Create a simple web application or API for users to upload images and receive disease diagnoses.

## License

This project is licensed under the [MIT License](LICENSE).

