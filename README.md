# Retail Profitability Analysis

An exploratory data analysis project using the Sample Superstore dataset to understand sales performance, profit margins, discount patterns, and regional profitability.

The primary deliverable is the Jupyter notebook: **[`retail_profitability_analysis.ipynb`](retail_profitability_analysis.ipynb)**.

## Project objective

Analyze retail transactions to identify product categories, sub-categories, and regions associated with stronger or weaker profitability. The analysis also explores how discount levels and delivery time relate to business performance.

## Dataset

- **Dataset:** Sample Superstore
- **Records:** 9,994 transactions
- **Columns:** 21 original fields
- **Coverage:** Orders, customers, products, regions, sales, discounts, and profit
- **Availability:** The CSV is not included in this repository. Place `Sample - Superstore.csv` in the project root before running the notebook.

## Tools used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Statsmodels (for the exploratory VIF check)

## Analysis performed

1. Loaded and inspected the dataset, including data types and missing values.
2. Converted order and shipping dates and derived delivery duration.
3. Compared sales and profit across categories, sub-categories, and regions.
4. Examined discount and profit patterns for selected sub-categories.
5. Analyzed monthly sales, profit, and profit-margin trends.
6. Calculated category and regional profit margins.
7. Explored a linear regression model as an additional learning exercise.

## Selected findings

The figures below are calculated from the notebook's analysis:

| Analysis | Finding |
|---|---:|
| Total sales | $2.30 million |
| Total profit | $286,397 |
| Furniture profit margin | 2.49% |
| Office Supplies profit margin | 17.04% |
| Technology profit margin | 17.40% |
| Tables profit | -$17,725 |
| Tables profit margin | -8.56% |
| Bookcases profit margin | -3.02% |
| Central-region Furniture profit margin | -1.75% |
| Discount-profit correlation | -0.219 |

### Business observations

- Furniture has a substantially lower overall profit margin than the other two categories.
- Tables and Bookcases are loss-making sub-categories in the analyzed dataset.
- Furniture in the Central region records a negative profit margin.
- The discount-profit correlation is negative, indicating an association between higher discounts and lower profit. Correlation alone does not establish causation.
- Monthly results show that sales growth does not necessarily translate into stable profit.

## Regression experiment and limitations

The notebook also includes an exploratory Linear Regression exercise. The initial model produced an R² of approximately **-0.645**. A second experiment, after filtering rows using the target variable's IQR, produced an R² of approximately **0.210** on the filtered data.

These results should be treated as a learning experiment, **not as evidence of a validated prediction model**. The current notebook filters observations using the target variable before the train-test split, so the evaluation does not represent a clean, unbiased test on the original data distribution. The modelling section should be revisited before using or presenting the model as a reliable predictor.

## Repository structure

```text
retail-profit-app/
├── retail_profitability_analysis.ipynb  # Main analysis notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── save_model.py                        # Earlier model-training prototype
├── app.py                               # Earlier Flask API prototype
├── streamlit_app.py                     # Earlier Streamlit UI prototype
├── retail_model.pkl                     # Saved prototype model
├── feature_columns.pkl                  # Prototype feature-column metadata
└── model_meta.pkl                       # Prototype dropdown metadata
```

The Flask/Streamlit prediction files are retained as an earlier prototype. The project is presented primarily as a data analysis project; no live deployment is currently provided or claimed.

## Run the analysis locally

1. Clone the repository:

   ```bash
   git clone https://github.com/SabeerKhan03/retail-profit-app.git
   cd retail-profit-app
   ```

2. Create and activate a virtual environment (recommended).

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset file named `Sample - Superstore.csv` in the repository root.

5. Open `retail_profitability_analysis.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run the cells from top to bottom.

## Author

**Sabeer Khan**  
[GitHub Profile](https://github.com/SabeerKhan03)
