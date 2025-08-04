
#   Housing Data Analysis & Modeling

## Overview

This project explores the **California Housing Dataset** using data analysis and preprocessing techniques, focusing on preparing the dataset for robust machine learning modeling. The project demonstrates key concepts such as:

- **Stratified Sampling**
- **Handling skewed distributions**
- **Log transformation for normalization**
- **Custom Transformers for feature engineering**
- **Data visualization**
- **Experimental ML setup** (SVR, KNN, etc.)

The notebook walks through a hands-on approach to preparing real-world housing data for predictive modeling, making it a valuable resource for data scientists, ML engineers, and learners.

---

##  Dataset

The dataset used is the built-in **California Housing Dataset** from `sklearn.datasets`. It includes various demographic and housing-related features from California districts. Key features include:

- `MedInc`: Median income in block group
- `HouseAge`: Median house age
- `AveRooms`: Average rooms per household
- `AveOccup`: Average house occupancy
- `Latitude` & `Longitude`: Geographical info
- `MedHouseVal`: Median house value (target)

---

##  Objectives

- Explore and visualize the data
- Handle noisy or multimodal distributions
- Normalize skewed features using log transformations
- Use `StratifiedShuffleSplit` to preserve target distribution
- Lay groundwork for model training using pipelines and transformers
- Prepare for regression modeling with Support Vector Regressor, KNN, and feature selection techniques

---

##  Requirements

Install the necessary Python packages using:

```bash
pip install -r requirements.txt
```

Main libraries used:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

---

##  Code Structure

### 1.  Data Loading

```python
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
```

### 2.  Data Exploration

- `.head()`, `.info()`, `.describe()` used for structure insight
- Histograms plotted to understand distributions
- Identification of skewed features

### 3.  Handling Skewed Data

- Log transformation applied to positively skewed columns to make them more normally distributed
- Purpose: Improve performance and convergence of ML models
- Applied to features like `MedInc`, `AveRooms`, etc.

```python
df['MedInc'] = np.log1p(df['MedInc'])
```

### 4.  Stratified Sampling

Used **StratifiedShuffleSplit** to split the dataset while maintaining the distribution of the target (income category):

```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

This ensures fair train-test distribution, crucial in real-world skewed data.

### 5.  Experimental Design

Custom experiments and exercises designed:

- **Adding Noise Flags**: Using custom flags instead of modifying original values to preserve data integrity
- **Custom Transformers**:
  - Example: KNN-based transformer using lat-long to predict neighborhood median house value
- **Model Prep Exercises**:
  - SVR with linear & RBF kernels
  - GridSearchCV vs RandomizedSearchCV
  - Feature selection via `SelectFromModel`
  - Inverse transform logic in `StandardScalerClone`

### 6.  Visualizations

Histograms plotted for key features to understand:
- Normality
- Multimodality
- Impact of log-transform

```python
df.hist(bins=50, figsize=(15, 8))
plt.show()
```

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook
```

---

##  Example Insights

- Median income was highly skewed → log-transformed
- Stratified sampling yielded more consistent train/test splits
- Exercises explored scalable solutions like dimensionality reduction and model-based feature selection

---

## Limitations

- Full modeling pipelines (e.g., model evaluation) may not be covered in detail
- Assumes a clean dataset; further preprocessing may be needed for production
- External datasets not integrated (only built-in housing data)

---

##  Future Enhancements

- Add complete ML training pipelines (Random Forest, SVR, etc.)
- Include performance metrics (MAE, RMSE, R²)
- Deploy trained model with Flask or Streamlit
- Automate transformation steps with custom pipeline components

---

