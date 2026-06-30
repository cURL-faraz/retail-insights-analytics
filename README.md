# 🛒 Retail Sales Analytics: End-to-End Business Deep Dive

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-4C4C4C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

This project demonstrates a comprehensive, end-to-end data analytics workflow. The goal is to clean a messy retail sales dataset, perform exploratory data analysis (EDA), and extract actionable business insights through advanced visualization and customer segmentation. 

> [!TIP]
> **Executive TL;DR:** This analysis reveals that revenue growth should prioritize **customer acquisition in underperforming regions**, **protecting VIP relationships**, and **doubling down on Electronics**. Scroll down to see the data backing these strategic moves.

---

## 📊 Strategic Insights & Key Findings

Based on the final executive report (`reports/report.md`), here is what the data tells us about our business drivers. 

* **Electronics Dominates the Bottom Line:** Electronics is the primary driver of business performance. It generates the highest revenue by perfectly combining the largest order volume with the highest average transaction value.
    *(See: `reports/figures/q1_category_signature_heatmap.png`)*

* **The West Region is a Volume Play:** The West is our strongest-performing region, but its advantage comes from serving *more customers* and processing *more orders*, rather than higher-value purchases. Customer acquisition is a stronger growth lever here than increasing basket size.
    *(See: `reports/figures/q2_region_decomposition.png`)*

* **The Discount Paradox:** Customer satisfaction increases consistently with larger discounts, and this trend remains visible even within major segments (like Electronics and the West). Discounting does not appear to cheapen or harm the brand experience.
    *(See: `reports/figures/q3_discount_vs_rating.png`)*

* **Hyper-Concentrated Revenue (The VIP Effect):** Revenue is highly concentrated. A small number of electronic products generate most sales, and the **top 10% of customers contribute ~33% of total revenue**. VIP retention is strategically critical.
    *(See: `reports/figures/q5_customer_vs_revenue_share.png`)*

* **The June Rebound was Isolated:** The sharp sales recovery observed in June 2024 was driven almost entirely by a surge in Electronics orders rather than broad, balanced growth across all categories.
    *(See: `reports/figures/q6_june_rebound_decomposition.png`)*

> [!NOTE] 
> **Strategic Recommendation:** Prioritize investment in the Electronics category, develop targeted retention programs for high-value customers, and expand the successful high-volume regional sales strategies of the West to our lower-performing territories.

---

## 🛠️ Project Phases

### Phase 1: Data Cleaning (`01_data_cleaning.ipynb`)
* Handled missing values via group-based imputation and targeted row dropping.
* Corrected data types (datetime, numeric, categorical).
* Removed duplicates and fixed string inconsistencies.
* Detected and treated outliers using the IQR method.
* **Output:** Generated a clean dataset (`retail_sales_cleaned.csv`).

### Phase 2: Exploratory Data Analysis (`02_exploratory_analysis.ipynb`)
* Analyzed sales distributions by category and region.
* Investigated time-based trends, identifying peak sales months and recovery periods.
* **Output:** Built an 8-panel exploratory dashboard (`reports/figures/eda_dashboard.png`).

### Phase 3: Business Deep Dive & Reporting (`03_business_deep_dive.ipynb`)
* Designed polished, custom visualizations using a modular plotting script (`src/plots.py`).
* Deconstructed regional leadership and analyzed pricing psychology.
* Segmented the customer base to map revenue share.
* **Output:** Generated publication-ready charts and an executive summary for management (`reports/report.md`).

---

## 📂 Directory Structure

```
data_analytics_project/
├── data/
│   ├── processed/
│   │   └── retail_sales_cleaned.csv
│   └── raw/
│       └── retail_sales.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_business_deep_dive.ipynb
├── reports/
│   ├── figures/
│   │   ├── eda_dashboard.png
│   │   ├── q1_category_signature_heatmap.png
│   │   ├── q2_region_decomposition.png
│   │   ├── q3_discount_vs_rating.png
│   │   ├── q4_rating_returns_scatter.png
│   │   ├── q5_customer_vs_revenue_share.png
│   │   └── q6_june_rebound_decomposition.png
│   └── report.md
├── src/
│   ├── data_cleaning.py
│   └── plots.py
├── requirements.txt
└── README.md
```

## 🚀 Installation & Usage

**1. Clone the repository**

```
git clone [https://github.com/cURL-faraz/retail-insights-analytics.git](https://github.com/cURL-faraz/retail-insights-analytics.git)
cd retail-insights-analytics
```

**2. Set up a virtual environment (Recommended)**

 Ensure you have Python 3.8+ installed, then create and activate your environment:

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**

```
pip install -r requirements.txt
```

**4. Execute the Analysis**

 Launch Jupyter Notebook to view the full cleaning pipeline, statistical summaries, and interactive visualizations. Run the notebooks sequentially (`01` ➔ `02` ➔ `03`) from the `notebooks/` directory:

```
jupyter notebook
```

**5. Outputs** Cleaned data will automatically populate in the `data/processed/` folder. All generated charts will be safely saved in the `reports/figures` directory.

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for your own portfolio or research purposes.

## 👨‍💻 Author

**Faraz Ameri** - [GitHub](https://www.google.com/search?q=https://github.com/cURL-faraz)

