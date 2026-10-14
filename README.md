# Tehran House Price Analysis and Prediction with PyTorch

An educational data science project that explores Tehran property listings and trains a small PyTorch neural network to estimate prices from area for properties in the Punak neighborhood.

## Project goals

- Clean and validate real-estate data.
- Explore numeric distributions and feature relationships.
- Compare Pearson, Spearman, and Kendall correlations.
- Train a regression neural network using PyTorch.
- Select the best training epoch using validation loss.
- Evaluate held-out predictions with MAE, RMSE, and R².
- Compare the neural network against a median-price baseline.

## Repository structure

```text
house-price-prediction/
├── data/
│   ├── raw/
│   │   └── housePrice.csv
│   └── processed/
├── notebooks/
│   └── house_price_analysis_pytorch.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Python 3.10 or newer is recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd house-price-prediction

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cd notebooks
jupyter lab
```

Open `house_price_analysis_pytorch.ipynb` and run the cells from top to bottom.

The notebook automatically selects CUDA when PyTorch detects a compatible GPU; otherwise it uses the CPU.

## Dataset and cleaning

The supplied CSV contains 3,479 rows and columns including `Area`, `Room`, `Parking`, `Warehouse`, `Elevator`, `Address`, `Price`, and `Price(USD)`.

The notebook converts area and price fields to numeric values, flags implausible area values, removes rows missing required fields, and removes exact duplicates. It then focuses on the Punak neighborhood for a small, interpretable modeling task.

The 20–1,000 m² area range is an educational assumption, not a universal real-estate rule. Review excluded records and adjust the rule only when justified by data or domain knowledge.

**Before publishing this repository publicly, verify the dataset's source, redistribution permission, and license.** The source and currency-conversion method were not provided with the files.

## Modeling approach

- **Input:** `Area`
- **Target:** `Price(USD)`
- **Model:** small feed-forward neural network built with PyTorch
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Validation:** used for best-checkpoint selection and early stopping
- **Final evaluation:** held-out test set, with MAE, RMSE, and R²
- **Baseline:** median price calculated from the training data

No performance claim is made in advance; run the notebook to see the results.

## Limitations

This is a learning project, not a professional valuation tool. Using area alone ignores many factors that influence property prices. Random splitting also does not establish how well the model will predict future market prices.

## Technologies

- Python
- PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (splitting and evaluation metrics)
- JupyterLab
