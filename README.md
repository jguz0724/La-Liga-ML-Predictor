La Liga Match Predictor: Machine Learning vs. Vegas Odds

This repository contains a Python-based predictive pipeline that forecasts Spanish La Liga match outcomes (Home Win, Away Win, Draw). The primary objective of this project is to benchmark machine learning classification models against the multi-billion-dollar sports betting industry (Vegas odds).

Project Overview:
Predicting soccer matches is notoriously difficult due to the low-scoring nature of the sport. To evaluate true predictive value, this project establishes strict benchmarks:
  -The Naive Baseline: Always predicting a Home Win yields 48.95% accuracy.
  -The Vegas Benchmark: Always picking the lowest bookmaker odds (Bet365 favorite) yields 55.00% accuracy.

Feature Engineering & Methodology:
To prevent data leakage, the models were trained exclusively on pre-match data. The dataset (SP1.csv) was enhanced using pandas to calculate 3-game rolling averages to capture team momentum:
  -Home_Goals_Form & Away_Goals_Form
  -Home_Shots_Target_Form & Away_Shots_Target_Form

Models & Results:
After performing an 80/20 train-test split to ensure blind testing, the models achieved the following accuracy:
  -Random Forest Classifier: 52.38%
  -XGBoost Classifier: 50.79%

While the models fall just short of the 55% Vegas benchmark, they successfully close the gap on the sportsbooks using only lightweight engineered features, without access to proprietary injury, weather, or referee data.

Key Insights:
  Extracting feature importance from the XGBoost model revealed that while Vegas odds (B365H, B365A) carry the heaviest predictive weight, a team's recent offensive momentum (Home_Goals_Form) is a stronger predictor of the match outcome than the baseline probability of a draw (B365D).

Tech Stack:
  -Language: Python
  -Libraries: pandas, scikit-learn, xgboost, matplotlib
