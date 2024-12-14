# California-Housing-Price-Prediction

## Project Synopsis
This Capstone Project explores the application of machine learning algorithms to predict housing prices in California, utilizing the California Housing dataset. The project focuses on understanding how various factors, such as geographic location, population density, proximity to key amenities, median household income, and other socioeconomic and environmental variables, influence housing prices. BVarious regression models were used, including Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, XGBoost, Lasso, and Support Vector Regression. These models were evaluated using metrics like MAPE, RMSE, MAE, and R² to compare their accuracy and performance. The project highlights the potential of machine learning to uncover key price drivers and support data-driven decision-making in real estate.

## Context
The California housing market is a complex and dynamic sector influenced by a multitude of factors such as location, socioeconomic conditions, and environmental aspects. The dataset under analysis provides information on housing prices, along with other features like the number of rooms, population, income levels, and geographical data (latitude and longitude). Predicting housing prices based on these factors is critical for understanding market trends and supporting stakeholders such as real estate agents, policymakers, and potential homeowners.

## Project Goals
1. Analyze the dataset to understand the key factors affecting housing prices
2. Develop regression models to predict housing prices with high accuracy
3. Evaluate and compare the performance of different machine learning models

## Conclusion
1. From the six machine learning models tested, the process was narrowed down to three top-performing models, ultimately identifying **Random Forest** as the best model for predicting California house prices. The model achieved an R² of **0.7521**, RMSE of **47,546.94**, MAE of **33,369**, and MAPE of **20.30%**, showcasing its accuracy and reliability in price estimation.
2. The analysis highlighted the importance of a comprehensive approach, including data cleaning, EDA, and data preprocessing, to ensure data quality and usability. Additionally, hyperparameter tuning played a critical role in significantly improving the model’s performance, underlining its necessity in building effective predictive models.
3. Among the 13 features analyzed, **median income** emerged as the most significant factor influencing housing prices. This finding highlights the dominant role of *economic*, *demographic*, and *geographical* factors in driving housing values. This insight is crucial for stakeholders focusing on market dynamics and investment strategies.

## Prediction Apps
Visit the app [[here]](https://california-housing-price-prediction.streamlit.app/).

