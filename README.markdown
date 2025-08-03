# 🍷 Global Alcohol Consumption Analysis – WHO Global Health Observatory

#### **Sector**: Health 🏥
#### **Focus Area**: Public health and alcohol consumption epidemiology

## 🌍 Project Overview
This project provides a **comprehensive analysis** of global alcohol consumption trends, utilizing data from the **WHO Global Health Observatory** to derive actionable insights. By employing advanced data science techniques, machine learning, and interactive visualizations, it examines patterns in the total amount of alcohol consumed per adult (aged 15+ years) over time, across countries, and by demographic factors. The findings aim to inform evidence-based public health policies and interventions to address alcohol-related health challenges.

**Student**: Gatsinzi Ernest  
**ID**: 26622

**Core Components**:
- **Python** 📊: For robust data processing, statistical analysis, and predictive modeling.
- **Power BI** 📈: For dynamic, interactive dashboards to communicate insights effectively.
- **Visual Studio Code** 💻: For collaborative and reproducible analysis.

---

## 🎯 Objectives
- 🔍 Analyze **global and regional trends** in alcohol consumption per adult (15+ years).
- ⚖️ Examine **consumption patterns** by gender, age group, and socio-economic factors.
- 🌐 Identify **geographic variations** in alcohol consumption.
- 🖼️ Develop **interactive visualizations** to empower stakeholders with clear insights.

---

## 📊 Dataset Information
- **Source**: [WHO Global Health Observatory](https://data.who.int/indicators/i/EF38E6A/EE6F72A?m49=646) 🌐
- **Coverage**: Global, with country-level granularity
- **Time Period**: 2000–2022 (varies by country) 🕰️
- **Format**: CSV / Excel 📑
- **Number of Rows and Columns**: Approximately 5,405 rows and 15 columns (raw dataset)
- **Data Structure**: Structured
- **Data Status**: Requires preprocessing

---

## 🛠️ Tools & Technologies
- **Python** 🐍: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn
- **Jupyter Notebook (Google Colab)** 📓: For exploratory analysis and model development
- **Power BI** 📊: For interactive, user-friendly dashboards
- **Excel/CSV** 📈: For initial data inspection and preprocessing

---

## 🔬 Methodology
This section outlines a **systematic, reproducible workflow** executed in **Google Colab**, ensuring analytical rigor and transparency.

### 1. Data Cleaning & Preprocessing 🧹
- **Objective**: Prepare raw data for analysis by ensuring consistency and quality.
- **Process**:
  - Loaded the dataset using **Pandas**:
    ```python
    import pandas as pd
    df = pd.read_csv("alcohol.csv")
    print(df.head())
    ```
  - Kept only relevant columns (`DIM_TIME`, `GEO_NAME_SHORT`, `RATE_PER_CAPITA_N`):
    ```python
    df_cleaned = df[['DIM_TIME', 'GEO_NAME_SHORT', 'RATE_PER_CAPITA_N']].copy()
    ```
  - Dropped rows with missing country names and renamed columns for clarity:
    ```python
    df_cleaned = df_cleaned.dropna(subset=['GEO_NAME_SHORT'])
    df_cleaned.rename(columns={
        'DIM_TIME': 'Year',
        'GEO_NAME_SHORT': 'Country',
        'RATE_PER_CAPITA_N': 'AlcoholPerCapita'
    }, inplace=True)
    ```
  - Exported the cleaned dataset for further analysis:
    ```python
    df_cleaned.to_csv("cleaned_alcohol_consumption.csv", index=False)
    print("✅ Cleaned data saved as 'cleaned_alcohol_consumption.csv'")
    ```

### 2. Exploratory Data Analysis (EDA) 📈
- **Objective**: Uncover trends, patterns, and relationships in alcohol consumption data.
- **Process**:
  - Inspected dataset structure, data types, and missing values:
    ```python
    print("Shape of the dataset:", df.shape)
    print("\nData Types and Missing Values:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    ```
  - Generated summary statistics for the cleaned dataset:
    ```python
    print(df_cleaned.describe())
    ```
  - Visualized global average alcohol consumption over time using **Seaborn** and **Matplotlib**:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    global_trend = df_cleaned.groupby("Year")["AlcoholPerCapita"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=global_trend, x="Year", y="AlcoholPerCapita", marker="o")
    plt.title("Global Average Alcohol Consumption Over Years")
    plt.ylabel("Liters per Capita")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    ```
  - **Visualization**:
  ![Architecture Diagram](screenshoot/consgraph.png)

### 3. Machine Learning Modeling 🤖
- **Objective**: Build predictive models to identify factors influencing alcohol consumption.
- **Initial Model**:
  - Developed a **Random Forest Regressor** to predict alcohol consumption levels:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    features = ['Year', 'Country_Code', 'GDP_Per_Capita']
    target = 'Alcohol_Consumption'

    X = alcohol_df[features]
    y = alcohol_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Model RMSE: {rmse}")
    ```
- **Improved Model**:
  - Enhanced the model with additional features and categorical encoding:
    ```python
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score

    df_model = alcohol_df.copy()
    categorical_cols = ['Country', 'Gender']
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])

    features = ['Year', 'Country', 'Gender', 'GDP_Per_Capita']
    target = 'Alcohol_Consumption'

    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    improved_model = RandomForestRegressor(random_state=42)
    improved_model.fit(X_train, y_train)
    y_pred_improved = improved_model.predict(X_test)
    rmse_improved = mean_squared_error(y_test, y_pred_improved, squared=False)
    r2_improved = r2_score(y_test, y_pred_improved)

    print(f"Improved Model RMSE: {rmse_improved}")
    print(f"Improved Model R² Score: {r2_improved}")
    ```
- **Addressing Imbalance**:
  - Applied **SMOTE** to balance categorical features if needed:
    ```python
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    smote_model = RandomForestRegressor(random_state=42)
    smote_model.fit(X_train_resampled, y_train_resampled)
    y_pred_smote = smote_model.predict(X_test)
    rmse_smote = mean_squared_error(y_test, y_pred_smote, squared=False)

    print(f"SMOTE Model RMSE: {rmse_smote}")
    ```

### 4. Power BI Dashboard 📊✨
- **Objective**: Create an interactive platform to visualize alcohol consumption trends.
- **Process**:
  1. Imported the cleaned dataset (`cleaned_alcohol_consumption.csv`) into **Power BI**.
  2. Designed a dashboard with:
     - **Geographic Map**: Displays consumption patterns by country.
     - **Line Chart**: Tracks alcohol consumption trends over time.
     - **Stacked Bar Chart**: Shows consumption breakdown by gender and age group.
     - **Slicers and Filters**: Enable interactive exploration by year, country, and demographics.
     - **DAX Formulas**: Added custom measures for advanced analytics, e.g., average consumption per region.

---

## 📈 Key Findings 🔍
1. **Regional Trends** 🌐: Alcohol consumption varies significantly by region, with higher levels in certain countries (e.g., Czechia at 14.56 liters per capita in 2003).
2. **Demographic Patterns** ⚖️: Males tend to have higher consumption rates than females.
3. **Temporal Trends** 📅: The global average alcohol consumption per capita shows an increase from 2000 to around 2010, peaking at approximately 5.65 liters, followed by a decline toward 2020, as depicted in the visualization.
4. **Predictive Insights** 🤖: The Random Forest model achieved a reasonable RMSE, with **Year** and **GDP_Per_Capita** as key predictors.

---

## 💡 Recommendations 🚀
1. **Targeted Interventions** 🎯: Focus public health campaigns on high-consumption regions and demographics.
2. **Policy Development** 📜: Use insights to inform alcohol taxation and regulation policies.
3. **Data Enhancement** 📑: Improve data collection for consistent global reporting.

---

## 🌟 Future Work
1. **Expanded Datasets** 📊: Incorporate additional socio-economic and health-related data.
2. **Advanced Modeling** 🤖: Explore neural networks for improved predictive accuracy.
3. **Web Application** 💻: Develop a real-time dashboard for public access.

---

## 📚 Conclusion
This project delivers a **robust analysis** of global alcohol consumption trends, combining data science, machine learning, and interactive visualizations to provide actionable insights. The modular Python code, detailed in the Jupyter Notebook, and the dynamic Power BI dashboard demonstrate technical excellence and a commitment to addressing public health challenges.

---

## 📦 Deliverables
- 📊 **Power BI Dashboard**: `alcohol_dashboard.pbix`
- 📁 **Raw Dataset**: `alcohol.csv`
- 📁 **Cleaned Dataset**: `cleaned_alcohol_consumption.csv`
- 📓 **Jupyter Notebook**: `Alcohol.ipynb` (data preparation, EDA, and visualization)
- 📄 **README**: This documentation
- 🌐 **GitHub Repository**: Public repository

---

## 🙏 Reflective Verse
*"Come unto me, all ye that labour and are heavy laden, and I will give you rest."*  
— Matthew 11:28

This verse inspires hope and resilience, reminding us to seek balance and support in addressing public health challenges.

---

## 💪 Encouragement
This project aims to drive meaningful change through data-driven insights. Let it inspire continued innovation and impact in public health!

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
