## Climate Action Using Machine Learning for Sustainable Development ☁️ :- 

## 1. Introduction:
The primary objective of this project is to develop a machine learning
model that analyzes energy consumption patterns to recommend optimal energy usage that
minimizes carbon emissions. By identifying trends in energy consumption and emissions,
this project aims to contribute towards actionable strategies that support carbon footprint
reduction, aligning with global sustainability goals.
## **Motivation:** :
This project is inspired by the United Nations Sustainable Development Goals
(SDGs), especially the goal centered on climate action. As global energy demands rise, there
is an urgent need to adopt efficient energy practices that reduce carbon footprints and
foster sustainable resource management.

## 2. Data Collection:- 
**Data Source:** Kaggle Dataset
**Dataset Description:**
The dataset combines monthly data on energy consumption, renewable energy production,
and carbon emissions, helping to identify patterns that reduce emissions:
1. **Date:** Month and year of each record.
2. **Energy Consumption:** Residential, Commercial, Industrial: Energy use in different
sectors.
3. **Renewable Production (MWh):** Solar, Wind: Renewable energy generation metrics.
4. **Carbon Emissions (metric tons):** Coal, Natural Gas, Oil: Emissions from each energy
source.

## 3.Exploratory Data Analysis (EDA): 
- **Distribution and Outliers:** Histograms revealed general distribution trends, while box plots
identified outliers, especially in industrial energy consumption and emissions.
- **Correlation Analysis:** Scatter plots showed strong correlations between
commercial/industrial energy use and coal emissions, indicating these sectors as major
contributors

## 4. Data Preprocessing
- **Splitting Data:** The data was divided into training and testing sets to ensure that the model
could generalize to new, unseen data. Typically, 80% of the data was used for training, and
20% for testing.
- **Features and Target Variable:** Key energy consumption features (e.g., Residential,
Commercial, and Industrial energy usage) and renewable energy production data (e.g., Solar
and Wind production) were used as input features (X), while Coal Emissions was the target
variable (y).

## 5. Machine Learning Model Selection
- **Objective:** Choose a simple model to predict coal emissions based on energy
consumption data for baseline insights.
- **Model:** Linear Regression: Selected for its simplicity, interpretability, and ability to
identify linear relationships between energy consumption and emissions.
- **Evaluation Metrics:** Mean Squared Error (MSE) and R² Score were used to assess
model accuracy.
- **Scikit-Learn:** Easy implementation, variety of algorithms, and effective performance
metrics.

## 7. Results and Evaluation
- **Performance Metrics:** The model achieved an R² score of 0.78, explaining 78% of the
variance in emissions.
- **Prediction Accuracy:** A scatter plot showed that predictions closely matched actual
emissions, proving the model’s baseline effectiveness.

## 8. Conclusion and Future Work
The project demonstrates a practical approach to predicting coal emissions based on energy
consumption patterns. The model provides insights into how residential, commercial, and
industrial consumption impacts emissions, with a focus on identifying high-emission
sectors that could benefit from targeted optimization strategies.

**Future Work:** Experimenting with time series or more complex machine learning models
like Random Forest or Gradient Boosting and Adding regional, seasonal, and policy-related
data to capture broader factors affecting emissions.

## 9. References
- Kaggle Dataset
- Scikit-Learn Documentation and Google Colaboratory
