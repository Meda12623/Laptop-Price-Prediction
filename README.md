# Laptop Price Prediction

A complete machine-learning project and web app that predicts laptop prices from technical specifications. The repository includes preprocessing, model training (ensemble), model persistence, and a Flask-based frontend built with HTML, CSS, and JavaScript.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Feature Engineering](#feature-engineering)
* [Modeling & Pipeline](#modeling--pipeline)
* [Training & Saving Model](#training--saving-model)
* [Flask Web App](#flask-web-app)
* [How to Run Locally (Development)](#how-to-run-locally-development)
* [How to Run on Google Colab](#how-to-run-on-google-colab)
* [Evaluation Metrics & Example Results](#evaluation-metrics--example-results)
* [Troubleshooting & Common Errors](#troubleshooting--common-errors)
* [Project Structure (Suggested)](#project-structure-suggested)
* [requirements.txt (Suggested)](#requirementstxt-suggested)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements & References](#acknowledgements--references)

---

## Project Overview

This repository contains a machine learning pipeline to predict the price of laptops using their specifications (brand, CPU, RAM, screen size, storage, GPU, OS, etc.). It also contains a Flask web application that accepts user input from an HTML form and returns a predicted price.

## Features

* Clean and documented preprocessing for numerical and categorical features.
* Encoding strategy: One-Hot Encoding for low-cardinality categories and Binary (or Base-N) encoding for high-cardinality categories.
* Scaling numeric features with MinMaxScaler.
* Ensemble model using a VotingRegressor (XGBoost + GradientBoosting + RandomForest + DecisionTree by default).
* Option to transform the target with `log10` for improved regression stability.
* Model persistence via `joblib` for fast loading in Flask.
* Frontend implemented with **HTML, CSS, JavaScript** for a responsive form and result display.

## Tech Stack

* Python 3.8+
* pandas, numpy
* scikit-learn (Pipeline, ColumnTransformer, VotingRegressor)
* XGBoost, GradientBoosting, RandomForest
* category\_encoders (BinaryEncoder) for high-cardinality categorical encoding
* joblib for model persistence
* Flask for serving the web app
* HTML / CSS / JavaScript for frontend

## Dataset

Place your dataset CSV in the repository (e.g. `laptop_price.csv`) or load it in Colab. Typical important columns used in this project:

* `Price_euros` (original price in euros)
* `Price_LE` (converted to local currency if needed)
* `Inches`, `Ram`, `Weight`, `Touchscreen`, `IPS`, `Memory_GB`, etc.
* Categorical columns: `Company`, `Product`, `TypeName`, `ScreenResolution`, `Cpu`, `Gpu`, `OpSys`, `Storage_Type`, ...

> Note: adapt column names in the code to match your dataset.

## Data Preprocessing

Recommended steps (code examples are in the repository):

1. **Load data** with `pandas.read_csv()` and inspect `df.info()` and `df.describe()`.
2. **Handle missing values**: drop or impute depending on the column.
3. **Target transformation**: if `Price` distribution is skewed, use `np.log10(price)` to stabilize variance.
4. **Numeric scaling**: use `MinMaxScaler` for numeric features.
5. **Categorical encoding strategy**:

   * If `n_unique < 7` → use `OneHotEncoder(drop='first')`.
   * If `n_unique >= 7` → use `BinaryEncoder` (from `category_encoders`) or another high-cardinality encoder.
6. **ColumnTransformer** to apply numeric and categorical transformers and keep the pipeline consistent for training and inference.

## Feature Engineering

* Extract numeric parts from text columns when necessary (e.g., screen resolution parsing).
* Create boolean flags (e.g., `has_ssd`, `is_touchscreen`).
* Consider polynomial or interaction features only if they improve validation metrics.

## Modeling & Pipeline

Example pipeline used in this project (high-level):

1. `ColumnTransformer` with numeric scaler + categorical encoders.
2. `VotingRegressor` wrapping base regressors: `XGBRegressor`, `GradientBoostingRegressor`, `RandomForestRegressor`, `DecisionTreeRegressor`.
3. Wrap preprocessing + regressor inside a `Pipeline` so the same transformations apply at predict time.

This ensures reproducibility and prevents data leakage.

## Training & Saving Model

1. Split into `X` and `y` (optionally apply `np.log10` on `y`).
2. `train_test_split(...)` with a fixed `random_state`.
3. Fit the pipeline: `pipeline.fit(X_train, y_train)`.
4. Evaluate on the test set and save the final pipeline (preprocessing + model) using `joblib.dump(pipeline, 'laptop_price_model.pkl')`.

When saving, keep in mind that `joblib.load()` expects the same Python environment (same library versions) to avoid compatibility issues.

## Flask Web App

Recommended endpoints and structure:

* `GET /` → serve HTML form where user selects or enters laptop specs.
* `POST /predict` → receive form data, preprocess (use the same pipeline), predict, and return price.

Typical `app.py` snippet to load model and predict:

```python
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('laptop_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # parse input into DataFrame row, e.g. data = pd.DataFrame([payload])
    pred = model.predict(data)
    # if model used log10 on target, apply inverse: 10**pred
    return render_template('index.html', prediction=pred_value)

if __name__ == '__main__':
    app.run(debug=True)
```

Make sure `laptop_price_model.pkl` is in the same folder as `app.py` or provide the absolute path.

## How to Run Locally (Development)

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Start Flask app:

```bash
python app.py
```

4. Open browser at `http://127.0.0.1:5000`.

## How to Run on Google Colab

1. Upload your CSV using `from google.colab import files; uploaded = files.upload()`.
2. Install missing packages (e.g., `!pip install category_encoders xgboost joblib`).
3. Train model and save with `joblib.dump()` then download the file or save to Google Drive.

## Evaluation Metrics & Example Results

Use these metrics for regression evaluation:

* **MAE** (Mean Absolute Error)
* **MSE** (Mean Squared Error)
* **RMSE** (Root Mean Squared Error)
* **R²** (Coefficient of Determination)

Example (from experiments):

```
Voting Regressor Results:
MAE: 0.0709
MSE: 0.0093
RMSE: 0.0967
R²: 0.8602
Train R²: 0.9824
Test R²: 0.8602
```

If you used `log10` transform on the target, compute metrics on the transformed scale or invert predictions before computing metrics depending on the interpretation you want.

## Troubleshooting & Common Errors

* **FileNotFoundError: laptop\_price\_model.pkl not found**

  * Put the file in the same folder as `app.py` or use the absolute path when loading with `joblib.load()`.

* **ModuleNotFoundError: category\_encoders**

  * Install it: `pip install category_encoders` (in Colab: `!pip install category_encoders`).

* **ValueError: continuous is not supported (classification\_report)**

  * `classification_report` is for classification. For regression use `mean_absolute_error`, `mean_squared_error`, `r2_score`.

* **KeyError when selecting columns**

  * Ensure you select columns from the same DataFrame used for feature selection (e.g., use `X.columns[sfs.get_support()]` if SFS fitted on `X`).

* **IndexError: boolean index did not match indexed array**

  * Make sure the boolean mask length equals the number of columns you index (mask from `sfs.get_support()` must match DataFrame columns used when fitting SFS).

## Project Structure (Suggested)

```
project-root/
├── app.py
├── laptop_price_model.pkl
├── laptop_price.csv
├── notebooks/
│   └── training_and_experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
├── requirements.txt
└── README.md
```

## requirements.txt (Suggested)

```
pandas
numpy
scikit-learn
xgboost
category_encoders
joblib
flask
seaborn
matplotlib
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make changes and test locally
4. Create a pull request with a clear description

## License

Choose a license (e.g., MIT) and put the LICENSE file in the repo.

## Acknowledgements & References

* scikit-learn (Pipeline, VotingRegressor, ColumnTransformer, metrics) — see official docs
* XGBoost (XGBRegressor)
* Flask (web app)
* pandas, numpy
* category\_encoders (BinaryEncoder)
* joblib (persistence)

---

*If you want, I can:*

* export this README as `README.md` in the repo for you,
* translate it to Arabic,
* shorten it to a one-page summary,
* generate a `requirements.txt` file or `app.py` template.
  ## 📬 Contact Me  

### Mina Ibrahim  
[![Mail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:minaibrahim365@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mina-ibrahim-ab7472313/)  
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/minaibrahim22)  


