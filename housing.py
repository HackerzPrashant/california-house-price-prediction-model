# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler , OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# import numpy as np


# data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\MachineLearning\housing.csv")
# #print(data.isnull().sum()) #total bedroom has 207 null
# # data.hist(bins=50, figsize=(20,15))
# # plt.show()
# data["rooms_per_household"] = data["total_rooms"] / data["households"]
# data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
# data["population_per_household"] = data["population"] / data["households"]

# x= data.drop(columns="median_house_value", axis=1) 
# z= data.drop("ocean_proximity", axis=1)
# y = data["median_house_value"]

# # print(z.corr()["median_house_value"].sort_values())

# x_train, x_test, y_train, y_test = train_test_split(x ,y , test_size=0.2,random_state=42)
# # print(x_test.shape)
# # sns.scatterplot(x= x_train["latitude"] , y= x_train["longitude"], alpha=0.1)
# # plt.show()
# # for col in z.columns:
# #     sns.scatterplot(x= x_train[col] , y= y_train, alpha=0.1)
# #     plt.show()
# # Numerical columns (exclude categorical)
# num_col = x.drop(columns=["ocean_proximity"]).columns

# # Categorical columns
# cat_col = ["ocean_proximity"]

# # Pipelines
# num_pipe = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])

# cat_pipe = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore"))
# ])

# # ColumnTransformer
# full_pipeline = ColumnTransformer([
#     ("num", num_pipe, num_col),
#     ("cat", cat_pipe, cat_col)
# ])

# x_train_transformed = full_pipeline.fit_transform(x_train)
# x_test_transformed = full_pipeline.transform(x_test)

# # print(x_train_transformed.shape, x_test_transformed.shape)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import  mean_squared_error
# from sklearn.metrics import r2_score
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# # lr = LinearRegression()
# # lr.fit(x_train_transformed, y_train)
# tree1 = DecisionTreeRegressor()
# tree = RandomForestRegressor()
# tree.fit(x_train_transformed, y_train)
# y_pred = tree.predict(x_test_transformed)

# print(mean_squared_error(y_test, y_pred))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(rmse)
# accoracy = r2_score(y_test, y_pred)
# print(accoracy)
# cvs = cross_val_score(tree, x_train_transformed, y_train, scoring="r2", cv=5)
# print(cvs)

 
# # After training the RandomForestRegressor
# import pandas as pd

# # Example: Ask the user for inputs
# # user_data = {}
# # for col in num_col:
# #     value = float(input(f"Enter value for {col}: "))
# #     user_data[col] = value

# # # Ask categorical column
# # for col in cat_col:
# #     value = input(f"Enter category for {col} (e.g., <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND): ")
# #     user_data[col] = value

# # Convert to DataFrame
# # user_df = pd.DataFrame([user_data])

# # # Apply the same pipeline transformation
# # user_transformed = full_pipeline.transform(user_df)

# # Predict price
# # predicted_price = tree.predict(user_transformed)
# # print(f"\n🏠 Estimated House Price: ${predicted_price[0]:,.2f}")

# import pickle


# # Save model to file
# with open("house_price_model.pkl", "wb") as f:  # write in binary mode
#     pickle.dump(tree, f)  # dump model into the file

# print("Model saved successfully!")
# with open("house_price_model.pkl", "rb") as f:  # read in binary mode
#     loaded_model = pickle.load(f)

# print("Model loaded successfully!")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\MachineLearning\housing.csv")

# Feature engineering
data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

X = data.drop(columns="median_house_value", axis=1)
y = data["median_house_value"]

# Column types
num_col = X.drop(columns=["ocean_proximity"]).columns
cat_col = ["ocean_proximity"]

# Pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipe, num_col),
    ("cat", cat_pipe, cat_col)
])

# Combine preprocessing + model into ONE pipeline
full_model = Pipeline([
    ("preprocessing", full_pipeline),
    ("model", RandomForestRegressor())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
full_model.fit(X_train, y_train)

# Evaluate
y_pred = full_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R² Score:", r2_score(y_test, y_pred))
print("Cross-val Scores:", cross_val_score(full_model, X_train, y_train, scoring="r2", cv=5))

# Save the combined pipeline + model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(full_model, f)

print("✅ Model & preprocessing saved together in 'house_price_model.pkl'")
