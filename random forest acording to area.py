import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# LOAD CSV DATA
# df = pd.read_csv("dataset.csv")
def datacleaning(csv):
    df = pd.read_csv(csv)
    # <-- use your real dataset
    df['Crop'] = df['Crop'].str.strip().str.lower()
    df['Season'] = df['Season'].str.strip().str.lower()
    df['State'] = df['State'].str.strip().str.lower()

    # Clean text
    df.columns = df.columns.str.strip()

    # Remove invalid crops
    remove_crops = ['coconut', 'arecanut']
    df = df[~df['Crop'].isin(remove_crops)]

    # Remove zero area
    df = df[df['Area'] > 0]

    # Convert to per hectare
    df['Fertilizer_per_ha'] = df['Fertilizer'] / df['Area']
    df['Pesticide_per_ha'] = df['Pesticide'] / df['Area']

    # Remove unrealistic values
    df = df[
        (df['Yield'] > 0) & (df['Yield'] < 100) &
        (df['Fertilizer_per_ha'] < 500) &
        (df['Pesticide_per_ha'] < 50) &
        (df['Annual_Rainfall'] < 4000)
    ]

    # Drop useless columns
    df = df.drop(columns=['Fertilizer', 'Pesticide', 'Production', 'Area'])

    # Remove duplicates & nulls
    df = df.drop_duplicates().dropna()
    df.to_csv("dataset.csv")


def model():
    df = pd.read_csv("dataset.csv")

    # -----------------------------
    # 2. ENCODING (your method)
    # -----------------------------
    le_dict = {}

    categorical_cols = ['Crop', 'Season', 'State']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # -----------------------------
    # 3. FEATURES & TARGET
    # -----------------------------
    X = df[['Crop', 'Season', 'State',
            'Annual_Rainfall', 'Fertilizer_per_ha', 'Pesticide_per_ha']]

    y = df['Yield']

    # -----------------------------
    # 4. TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 5. MODEL
    # -----------------------------
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42
    )

    # -----------------------------
    # 6. TRAIN
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # 7. EVALUATION
    # -----------------------------
    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # -----------------------------
    # 8. PREDICTION (IMPORTANT FIX)
    # -----------------------------
    # You MUST encode input using SAME encoders

    sample = {
        'Crop': 'onion',
        'Season': 'rabi',
        'State': 'gujarat',
        'Annual_Rainfall': 800,
        'Fertilizer_per_ha': 120,
        'Pesticide_per_ha': 10
    }

    # Convert to DataFrame
    sample_df = pd.DataFrame([sample])

    # Apply same encoding
    for col in categorical_cols:
        le = le_dict[col]
        sample_df[col] = le.transform(sample_df[col])

    # Predict
    prediction = model.predict(sample_df)

    print(f"\n🌾 Predicted Yield: {prediction[0]:.2f} tons per hectare (t/ha)")

    # # -----------------------------
    # # 1. Generate 200 inputs
    # # -----------------------------
    # test_data = pd.DataFrame({
    #     'Crop': np.random.choice(le_dict['Crop'].classes_, 200),
    #     'Season': np.random.choice(le_dict['Season'].classes_, 200),
    #     'State': np.random.choice(le_dict['State'].classes_, 200),
    #     'Annual_Rainfall': np.random.randint(300, 2000, 200),
    #     'Fertilizer_per_ha': np.random.randint(20, 300, 200),
    #     'Pesticide_per_ha': np.random.randint(1, 20, 200)
    # })

    # # Keep a copy BEFORE encoding (for readability)
    # original_data = test_data.copy()

    # # -----------------------------
    # # 2. Encode
    # # -----------------------------
    # for col in ['Crop', 'Season', 'State']:
    #     test_data[col] = le_dict[col].transform(test_data[col])

    # # -----------------------------
    # # 3. Predict
    # # -----------------------------
    # pred = model.predict(test_data)

    # # -----------------------------
    # # 4. Combine input + output
    # # -----------------------------
    # original_data['Predicted_Yield_t_per_ha'] = pred

    # # -----------------------------
    # # 5. Save to CSV
    # # -----------------------------
    # original_data.to_csv("model_test_results.csv", index=False)

    # # -----------------------------
    # # 6. View sample
    # # -----------------------------
    # print(original_data.head())

# datacleaning()
model()
