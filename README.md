# Streamlit Diabetes Risk Predictor

A Streamlit web application that uses a trained machine-learning pipeline to estimate the probability of a positive diabetes outcome from eight patient characteristics.

## Live application

**[Open the deployed Streamlit application](https://streamlit-diabetes-predictor-1.onrender.com/)**

https://streamlit-diabetes-predictor-1.onrender.com/

The application is hosted as a Render web service. Because it uses Render's free tier, the service may take approximately one minute to wake up after a period of inactivity.

## Project overview

This project demonstrates how a trained machine-learning model can be integrated into an interactive Streamlit interface and deployed online.

The model was originally developed for a Flask deployment project. The same trained pipeline was reused here so that the project could focus on:

- Building an interactive interface with Streamlit
- Collecting and validating user input
- Generating model predictions and probabilities
- Displaying results clearly
- Deploying a Streamlit application on Render

## Input features

The model expects the following eight features:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin-fold thickness |
| `Insulin` | Serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |

## Model pipeline

The saved scikit-learn pipeline contains preprocessing and prediction steps.

### Preprocessing

The following variables may contain `0` values that represent missing measurements:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

A `SimpleImputer` replaces these zero values with the median learned from the training data. The other features pass through the preprocessing step unchanged.

Because preprocessing is stored inside the same pipeline as the model, the application applies exactly the same transformations that were used during model development.

### Classifier

The prediction model is a `RandomForestClassifier` configured with:

```text
n_estimators=300
max_depth=6
min_samples_leaf=4
random_state=42
```

The application reports:

- The predicted outcome class
- The estimated probability of the positive class
- A lower- or higher-likelihood message
- The submitted feature values

## Streamlit interface

The interface was developed entirely in Python with Streamlit. It includes:

- Numeric inputs organized into two columns
- A form that submits all values together
- A probability metric and progress bar
- Color-coded prediction messages
- An expandable table of submitted values
- A medical-use disclaimer

The trained pipeline is loaded with `joblib` and cached with `st.cache_resource`. Streamlit reruns the script when the user interacts with the application, while caching prevents the model from being repeatedly loaded from disk.

## Project structure

```text
streamlit-diabetes-predictor/
├── models/
│   └── diabetes_pipeline.pkl
├── src/
│   └── app.py
├── .python-version
├── requirements.txt
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/icuhekim/streamlit-diabetes-predictor.git
cd streamlit-diabetes-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run src/app.py
```

The application will usually open at:

```text
http://localhost:8501
```

## Render deployment

The application is deployed on Render as a Python web service.

### Build command

```bash
pip install -r requirements.txt
```

### Start command

```bash
streamlit run src/app.py --server.address 0.0.0.0 --server.port $PORT
```

### Health-check path

```text
/_stcore/health
```

Python 3.12 is specified in `.python-version` to maintain compatibility with the project's pinned dependencies.

## External resources

The following documentation was used during development and deployment:

- [Streamlit documentation](https://docs.streamlit.io/)
- [Render documentation](https://render.com/docs)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

## Limitations

This application is an educational demonstration and is not intended for diagnosis, screening, treatment decisions, or other clinical use.

The probability is a model estimate based on patterns in the training dataset. It is not equivalent to an individual patient's true probability of having or developing diabetes. Performance may differ in populations that are not adequately represented in the original data.

## Technologies

- Python 3.12
- Streamlit
- pandas
- scikit-learn
- joblib
- Render