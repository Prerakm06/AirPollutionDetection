# AirPollutionDetection

## Introduction
* AirPollutionDetection is a machine learning-powered system built to detect and predict **Nitrogen Dioxide (NO₂)** AQI levels with high accuracy.
* This project was developed during my **internship at AWS** alongside team members **Sophie Liu** and **Michael Zhang**.
* Our model achieved a remarkable **89% R² score**, leveraging historical environmental data.
* Built using **AWS SageMaker** and **Scikit-learn**, the system uses tested ML models to deliver reliable and real-time predictions.
* This project is for **educational and non-commercial use** only.

![Air Pollution Detection](Images)

## The Problem
* **NO₂** is a highly reactive pollutant with severe impacts on **human health** and the **environment**, including:
  * Respiratory inflammation
  * Hindered lung development
  * Acid rain and visibility degradation
* Accurate NO₂ detection is challenging but critical for public health.

## Our Solution
* We developed a **regression-based ML model** to predict NO₂ AQI levels using:
  * Historical air quality data (CO, SO₂, O₃)
  * Demographics (city, state, county)
  * AQI metrics (mean, 1st max value/hour)
* Applications include:
  * **Smart homes** (air purifier control)
  * **Public health alerts** (SMS/email systems)
  * **Urban planning** (real-time air quality maps)

## Dataset Overview
* **1.7+ million data samples**
* Covers **47 US states** from **2000–2016**
* **24 features** from Kaggle, including:
  * NO₂, CO, O₃, SO₂
  * Dates, times, AQI values
  * Geographic metadata

## Data Processing
* **Feature Deletion**: Removed units/duplicates
* **Label Encoding**: Applied to categorical fields
* **Imputation**: Filled missing values using **SimpleImputer**
* **Scaling**: Normalized with **StandardScaler**
* **Balancing**: Not required due to continuous dataset

## Model Testing and Evaluation

### ✅ KNeighbors Regressor
* **K=10** yielded the best results
* **Max R² Score**: **0.9428**
* **Training Time**: 176 seconds

### ✅ Decision Tree Regressor (Best Performing)
* **Max R² Score**: **0.99998**
* **Training Time**: 0.989 seconds
* **Max Depth**: 45

### ✅ Linear Regression
* **Max R² Score**: **1.00**
* **Training Time**: 2.49 seconds
* **Mean Absolute Error**: 6.72e-16

### ✅ MLP Regressor (Neural Network)
* **Max R² Score**: **0.99988**
* **Training Time**: 7.619 seconds

## Final Recommendation
* **Decision Tree Regressor** selected for deployment:
  * Near-perfect accuracy
  * Lowest training time
  * Easy interpretability and deployment

## Why AWS SageMaker?
* **Ease of Deployment**: Full-cycle ML platform from preprocessing to model monitoring
* **Cost-Efficient**: Used free-tier S3 storage and Jupyter notebooks
* **Seamless Integration**: Stored and retrieved datasets directly from AWS S3

## Software Setup
1. Install dependencies:
   ```sh
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```
2. Load the dataset into AWS SageMaker notebook or local Python environment.
3. Preprocess the dataset using label encoding, imputation, and scaling.
4. Train models using `sklearn.model_selection` and evaluate using `r2_score`, `mean_squared_error`.

## Code Overview
Here's a simplified snippet of our Decision Tree Regressor:

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# X, y are the processed features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeRegressor(max_depth=45)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("R² Score:", r2_score(y_test, predictions))
```

## Future Improvements
* Deploy REST API using **Flask** for real-time predictions
* Integrate SMS alert system using **Twilio**
* Expand dataset beyond 2016 using EPA APIs
* Incorporate **time-series analysis** with LSTM models

## Acknowledgments
* Huge thanks to **AWS** and **Delta Careers** for providing the tools and opportunity to explore machine learning.
* Special thanks to my teammates:
  * **Prerak Mahajan**
  * **Sophie Liu**
  * **Michael Zhang**

## Contact
* [Prerak Mahajan](https://www.linkedin.com/in/prerakmahajan/)
* [Michael Zhang](https://www.linkedin.com/in/michael-zhang-1954b3284/)
* [Sophie Liu](http://www.linkedin.com/in/sophie-liu-06a029323)
