# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, r2_score

# # ==============================
# # 🔁 MODE SELECTION
# # ==============================
# mode = input("Enter mode (1 = CSV, 2 = Manual): ")

# # ==============================
# # 📂 MODE 1: LOAD CSV DATA
# # ==============================
# if mode == "1":
#     df = pd.read_csv("your_dataset.csv")
#     df = df.dropna()

# # ==============================
# # ✍️ MODE 2: MANUAL DATA
# # ==============================
# elif mode == "2":
#     data = {
#         "Crop": ["Rice", "Wheat", "Maize", "Rice"],
#         "Season": ["Kharif", "Rabi", "Kharif", "Rabi"],
#         "State": ["Gujarat", "Punjab", "UP", "Bihar"],
#         "Rainfall": [200, 150, 180, 120],
#         "Fertilizer": [100, 80, 90, 70],
#         "Area": [50, 40, 45, 30],
#         "Yield": [60, 50, 55, 45]
#     }
    
#     df = pd.DataFrame(data)

# else:
#     print("Invalid mode selected")
#     exit()

# # ==============================
# # 🔤 ENCODING
# # ==============================
# le_dict = {}  # store encoders

# categorical_cols = ['Crop', 'Season', 'State']

# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     le_dict[col] = le  # save encoder

# # ==============================
# # 🎯 FEATURES & TARGET
# # ==============================
# X = df.drop("Yield", axis=1)
# y = df["Yield"]

# # ==============================
# # ✂️ TRAIN TEST SPLIT
# # ==============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ==============================
# # 🌳 MODEL
# # ==============================
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # ==============================
# # 📊 EVALUATION
# # ==============================
# y_pred = model.predict(X_test)

# print("\nModel Performance:")
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# # ==============================
# # 🔮 MANUAL PREDICTION (USER INPUT)
# # ==============================
# print("\n--- Predict New Data ---")

# crop = input("Enter Crop: ")
# season = input("Enter Season: ")
# state = input("Enter State: ")
# rainfall = float(input("Enter Rainfall: "))
# fertilizer = float(input("Enter Fertilizer: "))
# area = float(input("Enter Area: "))

# # Encode input using saved encoders
# crop = le_dict['Crop'].transform([crop])[0]
# season = le_dict['Season'].transform([season])[0]
# state = le_dict['State'].transform([state])[0]

# # Create input array
# input_data = [[crop, season, state, rainfall, fertilizer, area]]

# # Predict
# prediction = model.predict(input_data)

# print("\nPredicted Yield:", prediction[0])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
print("started...")
# ==============================
# 🔁 MODE SELECTION
# ==============================

# ==============================
# 📂 MODE 1: LOAD CSV DATA
# ==============================
df = pd.read_csv("crop_yield_a.csv")   # <-- use your real dataset
df['Crop'] = df['Crop'].str.strip().str.lower()
df['Season'] = df['Season'].str.strip().str.lower()
df['State'] = df['State'].str.strip().str.lower()
df = df.dropna()

# ==============================
# 🔤 ENCODING
# ==============================
le_dict = {}

categorical_cols = ['Crop', 'Season', 'State']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ==============================
# 🎯 FEATURES & TARGET
# ==============================
X = df.drop("Yield", axis=1)
y = df["Yield"]

# ==============================
# ✂️ TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🌳 MODEL
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 📊 EVALUATION
# ==============================
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ==============================
# 🔮 MANUAL PREDICTION
# ==============================
while True:
    try:
        print("\n--- Predict New Data ---")

        crop = input("Enter Crop (e.g., Rice): ").strip().lower()
        # year = int(input("Enter Year (e.g., 2020): "))
        season = input("Enter Season (Kharif/Rabi): ").strip().lower()
        state = input("Enter State: ").strip().lower()

        # area = float(input("Enter Area (in hectares): "))
        # production = float(input("Enter Production (in tons): "))
        # rainfall = float(input("Enter Annual Rainfall (in mm): "))
        # fertilizer = float(input("Enter Fertilizer (kg per hectare): "))
        # pesticide = float(input("Enter Pesticide (kg per hectare): "))



        # crop = "Rice".strip().lower()
        # season = "Kharif".strip().lower()
        # state = "Gujarat".strip().lower()
        year = 2020


        area = 50            # hectares
        production = 2500    # tons
        rainfall = 800       # mm
        fertilizer = 120     # kg/ha
        pesticide = 10       # kg/ha

        # Encode categorical inputs
        crop = le_dict['Crop'].transform([crop])[0]
        season = le_dict['Season'].transform([season])[0]
        state = le_dict['State'].transform([state])[0]


        # Create input
        input_data = [[crop, year, season, state, area, production, rainfall, fertilizer, pesticide]]

        # Predict
        prediction = model.predict(input_data)

        print(f"\n🌾 Predicted Yield: {prediction[0]:.2f} tons per hectare (t/ha)")
    except Exception as e:
        pass