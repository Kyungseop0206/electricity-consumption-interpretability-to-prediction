# Electricity Consumption Prediction: From Weather/Seasonal Interpretability to Enhanced Accuracy with Extended Features.
## [Part1: Interpretation] Weather/Seasonal Only Modeling with Parametric vs. Nonparametric Approaches
## [Part2: Prediction] Enhanced Prediction with Extended Features and Comparison of Nonparametric Models

## Key Results
### Part1: Interpretation
- Weather and Seasonality affects electricity consumption, but in a **nonlinear** way, making **nonparametric models outperform parametric GLMs**.
- **Temperature is the most influential feature**, while other weather features show weaker effects.
- The model's **$R^2$ remained around 0.67**, indicating that **weather alone cannot fully explain electricity consumption**.

### Part2: Prediction
| Model | Train RMSE | Test RMSE | $R^2$ |
| ----- | ----- | ----- | ----- |
| Only Weather/Seasonal Features |
| RandomForest Tuned | 0.302 | 0.568 | 0.674 |
| Extended Features |
| CatBoost | 0.053 | 0.115 | 0.987 |

- The best model (CatBoost) **increased $R^2$  by 46%** and **decreased RMSE by 79%** after adding state and lag features.
- Reduced overfitting, with a smaller gap between train and test RMSE.


## Project Overview
### Real World Motivation: Weather, Climate Change & Electricity Demand
- Weather is known to influence electricity consumption, so understading this relationship provides valuable insights for both climate and energy.
- Climate change is intensifying weather extremes, increasing the need to understand how these changes affect electricity consumption.
- Accurate electricity consumption predicting is important for grid stability and energy planning, helping utilities allocate resources efficiently and reduce the risk of electricity shortages.

### Key Research Question
- To what extent can U.S. electricity consumption be explained using only weather and seasonal features, and how do parametric vs. nonparametric models differ in interpretability and performance?
- Can predictive accuracy be further improved by incorporating additional features beyond weather and seasonality?

### Goal:
- Compare parametric (GLMs) and nonparametric models to examine linear vs. nonlinear relationships between weather and electricity consumption.
- Examine how much electricity consumption can be explained using only weather and seasonal features, and how each feature contributes.
- Improve prediction accuracy by adding additional features beyond weather and seasonality.


## Dataset Source
| Data | Source |
| ----- | ----- |
| Avg temp, wind, PMDI, precipitation | National Centers for Environmental Information (NCEI) |
| Residential Electricity Consumption | U.S. Energy Information Administration |


## Tech Stack
- **Data Processing:** Pandas, Numpy
- **Modeling:** Scikit-learn, RandomForest, CatBoost, XGBoost, statsmodels (GLMs)
- **Metrics:** RMSE, $R^2$
- **Visualization:** Matplotlib, Seaborn
- **Developmen Environment:** Google Colab


## Full Analysis Notebook
[part1_understanding_electricity_consumption.ipynb](./notebooks/part1_understanding_electricity_consumption.ipynb)

[part2_predicting_electricity_consumption.ipynb](./notebooks/part2_predicting_electricity_consumption.ipynb)


## Analysis Workflow
### Part1: Interpretation
#### 1. Data Explore & Preprocessing
- Cleaned and merged separate datasets by Year, Month, and State.

#### 2. Feature Engineering
- Handled missing values and created seasonal features.
- Converted electricity unit from MWh to GWh for interpretability and numerical stability.

#### 3. Exploratory Data Analysis (EDA)
- Analyzed electricity consumption monthly and yearly trends.
- Explored weather seasonality patterns.
- Checked linear vs. nonlinear relationships between weather and electricity consumption.
- Examined electricity consumption distribution for skewness and model suitability.

#### 4. Modeling & Evaluation
- Applied log transformation and z-socre standardization on the target variable.
- Parametric Models: Gaussian & Gamma Generalized Linear Model (GLM).
- Nonparametric Models: RandomForest with RandomizedSearchCV.
- Interpreted GLM summary and RF feature Importance.
- Evaluated Performance using RMSE and $R^2$.


### Part2: Prediction
#### 1. Feature Engineering & EDA
- Created 1 and 12 month Lag features for electricity consumption.
- Explored state level consumption patterns to capture regional differences.
- Examined autocorrelation to observe how past consumption influences current consumption.
- Checked correlation among all features, including target variables, to identify relationships.

#### 2. Modeling & Evaluation
- Compared RandomForest, XGBoost, and Catboost to identify the best performaing model.
- Applied RandomizedSearchCV for the hyperparameter tuning.
- Used Time Series Split to ensure time aware cross validation and prevent data leakage.
- Evaluated model performance using RMSE and $R^2$, the same metrics used in Part1.


## Key Insights & Discussion
### Insights
- Weather and seasonal features explain a meaningful portion of electricity consumption but cannot fully explain it, indicating that influence of broader social and structural factors, such as population, policy, or energy prices.
- Nonparametric models captured nonlinear relationships that GLM missied, showing how temperature sensitivity differs across states and seasons. These can help improve energy planning as climate patterns change, different regions respond to weather in different ways.
- Adding lag and state features significantly improved performance, capturing both temporal and regional patterns, key for state specific prediction and localized policy strategies.

### Challenges
**GLM Limitations**
- GLMs assume linear relationships, which did not hold for weather and electricity consumption patterns, leading to  unstable results.

**Limited Explainability of Tree based Models**
- Tree based models capture nonlinear relationships effectively, but they do not provide specific details of directional or coefficients.

**Lag and State Dominance Reduced Interpretability**
- Model relies on past consumption patterns or state specific characteristics, making it harder to interpret the direct effects of weather on electricity consumption.


## Future Improvement
**Extend Weather Feature Scope**
- Add more weather variables such as humidity, extreme weather events, and heatwave/coldwave indicators to better capture climate related consumption patterns.

**Improve Regional Feature Representation**
- Instead of using state identifiers, include state level characteristics (e.g. income, electricity price, climate zone) to explain why regions differ.