import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

#  Load the dataset
match_data = pd.read_csv("SP1.csv")

# Feature Engineering
match_data['Home_Shots_Target_Form'] = match_data.groupby('HomeTeam')['HST'].transform(lambda x: x.shift(1).rolling(3).mean())
match_data['Away_Shots_Target_Form'] = match_data.groupby('AwayTeam')['AST'].transform(lambda x: x.shift(1).rolling(3).mean())
match_data['Home_Goals_Form'] = match_data.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(3).mean())
match_data['Away_Goals_Form'] = match_data.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(3).mean())

features = [
    'B365H', 'B365D', 'B365A', 
    'Home_Shots_Target_Form', 'Away_Shots_Target_Form',
    'Home_Goals_Form', 'Away_Goals_Form'
]

match_data = match_data.dropna(subset=features + ['FTR'])

X = match_data[features]
y = LabelEncoder().fit_transform(match_data['FTR'])

# Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Generate Feature Importance Visualization
print("Generating Feature Importance Chart...")
importance = model.feature_importances_

# Sort features by importance for the chart
feature_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='#1f77b4')
plt.title('XGBoost Feature Importance: La Liga Match Prediction')
plt.xlabel('Importance Weight')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
