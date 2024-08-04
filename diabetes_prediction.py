import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("diabetes.csv")

# Basic data exploration
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.Outcome.value_counts())

# Train test split
X = df.drop("Outcome", axis="columns")
y = df.Outcome

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Train using standalone model
standalone_model = DecisionTreeClassifier()
standalone_scores = cross_val_score(standalone_model, X, y, cv=5)
standalone_mean_score = standalone_scores.mean()

# Train using Bagging
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)

bag_model.fit(X_train, y_train)
bagging_scores = cross_val_score(bag_model, X, y, cv=5)
bagging_mean_score = bagging_scores.mean()

# Train using Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=0)
rf_scores = cross_val_score(rf_model, X, y, cv=5)
rf_mean_score = rf_scores.mean()

# Evaluate models on the test set
print("Standalone Decision Tree test score:", standalone_model.fit(X_train, y_train).score(X_test, y_test))
print("Bagging Classifier test score:", bag_model.score(X_test, y_test))
print("Random Forest test score:", rf_model.fit(X_train, y_train).score(X_test, y_test))

# Visualize the results
models = ['Decision Tree', 'Bagging Classifier', 'Random Forest']
scores = [standalone_mean_score, bagging_mean_score, rf_mean_score]

plt.figure(figsize=(10, 6))
plt.bar(models, scores, color=['blue', 'green', 'orange'])
plt.xlabel('Model')
plt.ylabel('Cross-Validation Score')
plt.title('Comparison of Model Performance')
plt.ylim(0, 1)  # Assuming score is between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('model_comparison.png')  # Save the plot as an image file
plt.show()
