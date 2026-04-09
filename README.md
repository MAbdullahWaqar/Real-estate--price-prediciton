# 🏠 NeoHome — Bangalore Real Estate Price Predictor

## Overview

NeoHome is an end-to-end machine learning application that predicts residential property prices in Bangalore, India. It combines a scikit-learn regression model trained on real Kaggle housing data with a Flask REST API and a responsive browser-based frontend. Users enter property details — area, BHK count, bathrooms, and neighborhood — and receive an instant price estimate in Indian Rupees (Lakh). The project demonstrates the full ML-to-production pipeline: data cleaning, feature engineering, model selection, serialization, API serving, and UI consumption.

## Key Features

- **Instant price estimation** for 240+ Bangalore neighborhoods
- **Linear Regression model** selected via `GridSearchCV` over multiple algorithms (Linear Regression, Lasso, Decision Tree)
- **Flask REST API** with CORS support, serving predictions and location metadata
- **Interactive frontend** (vanilla JS + jQuery) with animated price reveal and 3D mouse-parallax effect
- **Pre-trained model artifacts** included — no training required to run the app
- **Data preprocessing pipeline** covering outlier removal, feature engineering, and one-hot encoding

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3, JavaScript (ES6) |
| ML Framework | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization (notebook) | Matplotlib |
| Backend | Flask |
| Frontend | HTML5, CSS3, jQuery 3.4 |
| Model Serialization | pickle |
| Fonts / Icons | Google Fonts (Orbitron, Exo 2), Font Awesome 6 |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/vanix056/Real-estate--price-prediciton.git
cd Real-estate--price-prediciton

# 2. Install Python dependencies
pip install flask numpy scikit-learn pandas matplotlib
```

> No `requirements.txt` is included in the repository. The packages above cover all runtime dependencies.

## Usage

### Start the Flask server

```bash
cd server
python server.py
```

The server starts at `http://127.0.0.1:5000` and loads the pre-trained model from `server/artifacts/`.

### Open the frontend

Open `client/index.html` directly in your browser (no build step required):

```bash
# macOS
open client/index.html

# Linux
xdg-open client/index.html

# Windows
start client/index.html
```

Fill in the property details and click **ESTIMATE PRICE** to get a prediction.

## Project Structure

```
Real-estate--price-prediciton/
├── client/
│   ├── index.html          # Single-page frontend UI
│   ├── script.js           # API calls and DOM interactions
│   └── styles.css          # Futuristic themed stylesheet
├── model/
│   ├── banglore_home_prices_final.ipynb   # Full ML pipeline notebook
│   ├── banglore_home_prices_model.pickle  # Trained model (source)
│   ├── bengaluru_house_prices.csv         # Raw Kaggle dataset
│   ├── bhp.csv                            # Cleaned intermediate dataset
│   └── columns.json                       # Feature column metadata
└── server/
    ├── server.py           # Flask application and API routes
    ├── util.py             # Model loading and prediction logic
    └── artifacts/
        ├── banglore_home_prices_model.pickle  # Deployed model
        └── columns.json                        # Deployed column metadata
```

## Model Architecture

| Attribute | Detail |
|---|---|
| Algorithm | Linear Regression (selected via GridSearchCV) |
| Input features | `total_sqft`, `bath`, `bhk`, one-hot encoded location (240 neighborhoods) |
| Output | Predicted price in Lakh Rupees |
| Validation | 5-fold `ShuffleSplit` cross-validation |
| Competitors evaluated | Lasso, Decision Tree Regressor |

## Dataset

- **Source:** [Bengaluru House Price Data — Kaggle](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)
- **Raw file:** `model/bengaluru_house_prices.csv`
- **Columns:** `area_type`, `availability`, `location`, `size`, `society`, `total_sqft`, `bath`, `balcony`, `price`

## Analysis Workflow

The notebook `model/banglore_home_prices_final.ipynb` walks through:

1. **Data loading** — Import CSV into a pandas DataFrame
2. **Data cleaning** — Drop irrelevant columns (`area_type`, `availability`, `society`, `balcony`), handle missing values
3. **Feature engineering** — Parse `size` into integer `bhk`; handle range-format `total_sqft` entries
4. **Outlier removal** — Filter rows with abnormally low sqft-per-BHK ratios and price outliers per location
5. **Dimensionality reduction** — Collapse locations with fewer than 10 listings into an `other` category
6. **One-hot encoding** — Expand 240 location categories into binary columns
7. **Model selection** — `GridSearchCV` with `ShuffleSplit(n_splits=5)` across three algorithms
8. **Serialization** — Export final model with `pickle` and column metadata to `columns.json`

## Visualizations

The notebook generates:

- Distribution plots for `total_sqft` and `price`
- Scatter plots of price vs. sqft per BHK category
- Bar charts highlighting price outliers by location

## API Routes

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/get_location_names` | Returns a list of all supported Bangalore neighborhoods |
| `POST` | `/predict_home_price` | Returns an estimated property price |

### `POST /predict_home_price`

**Request (form data):**

```
total_sqft=1200&bhk=2&bath=2&location=Whitefield
```

**Response:**

```json
{
  "estimated_price": 84.52
}
```

Price is expressed in **Lakh Rupees** (1 Lakh = ₹100,000).

## Evaluation Metrics

Model selection was performed using R² score via cross-validation:

- **Cross-validation:** 5-fold `ShuffleSplit`, `test_size=0.2`
- **Primary metric:** R² (coefficient of determination)
- Linear Regression consistently outperformed Lasso and Decision Tree on this dataset

## Configuration

No environment variables are required. The only configurable paths are hardcoded in `server/util.py`:

```python
# Artifact paths (relative to the server/ directory)
'./artifacts/columns.json'
'./artifacts/banglore_home_prices_model.pickle'
```

The frontend API base URL is set in `client/script.js`:

```javascript
var url = "http://127.0.0.1:5000";
```

Update this value if you deploy the Flask server to a remote host.

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`)
2. Commit changes with clear messages (`git commit -m "feat: add xyz"`)
3. Open a pull request with a description of what was changed and why
4. Ensure the Flask server starts cleanly and the frontend loads without errors before submitting

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Author

**vanix056**  
GitHub: [@vanix056](https://github.com/vanix056)
