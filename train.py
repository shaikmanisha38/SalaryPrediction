import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Load dataset
df = pd.read_csv('data/Salary_Dataset_with_Extra_Features.csv')

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Select features and target (now including Salary Reported)
features = ['Rating', 'Company Name', 'Job Title', 'Location',
            'Employment Status', 'Job Roles', 'Salaries Reported']
target = 'Salary'

X = df[features]
y = df[target]

# Encode categorical variables
label_encoders = {}
for col in ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numeric features (Rating and Salary Reported)
scaler = StandardScaler()
X[['Rating', 'Salaries Reported']] = scaler.fit_transform(X[['Rating', 'Salaries Reported']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure Model directory exists
os.makedirs('Model', exist_ok=True)

# Save model, encoders, and scaler
with open('Model/Software Industry Salary Prediction.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'label_encoders': label_encoders,
        'scaler': scaler
    }, f)

print("Model trained and saved successfully!")
