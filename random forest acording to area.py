import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
print("started...")

# LOAD CSV DATA
df = pd.read_csv("crop_yield_a.csv")   # <-- use your real dataset
df['Crop'] = df['Crop'].str.strip().str.lower()
df['Season'] = df['Season'].str.strip().str.lower()
df['State'] = df['State'].str.strip().str.lower()
df = df.dropna()

# ENCODING
le_dict = {}

categorical_cols = ['Crop', 'Season', 'State']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# FEATURES & TARGET
X = df.drop(["Yield", "Production"], axis=1)
y = df["Yield"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# EVALUATION
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# MANUAL PREDICTION
while True:
    try:
        print("\n--- Predict New Data ---")

        crop = input("Enter Crop (e.g., Rice): ").strip().lower()
        # year = int(input("Enter Year (e.g., 2020): "))
        season = input("Enter Season (Kharif/Rabi): ").strip().lower()
        state = input("Enter State: ").strip().lower()

        year = 2020

        area = 50            # hectares
        # production = 2500    # tons
        rainfall = 800       # mm
        fertilizer = 120     # kg/ha
        pesticide = 10       # kg/ha

        # Encode categorical inputs
        crop = le_dict['Crop'].transform([crop])[0]
        season = le_dict['Season'].transform([season])[0]
        state = le_dict['State'].transform([state])[0]


        # Create input
        input_data = [[crop, year, season, state, area, rainfall, fertilizer, pesticide]]

        # Predict
        prediction = model.predict(input_data)

        print(f"\n🌾 Predicted Yield: {prediction[0]:.2f} tons per hectare (t/ha)")
    except Exception as e:
        pass
