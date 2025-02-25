import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
 
# Step 1: Load Dataset
file_path = r"C:\Users\Admin\downloads\Market.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Step 2: Identify Numeric and Categorical Features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Step 3: Define Pipelines for Preprocessing
# Numerical Pipeline: Impute missing values and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Standardization
])

# Categorical Pipeline: Impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
])

# Step 4: Combine Pipelines Using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Step 5: Apply Transformation
df_processed = preprocessor.fit_transform(df)

# Convert transformed data into a DataFrame
df_transformed = pd.DataFrame(df_processed)

# Step 6: Save Processed Data
output_path = "processed_data.csv"
df_transformed.to_csv(output_path, index=False)

print("✅ Data preprocessing complete! Processed data saved as 'processed_data.csv'.")