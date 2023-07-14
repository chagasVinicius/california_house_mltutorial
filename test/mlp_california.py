from pycarol import Carol, Staging, ApiKeyAuth, Storage

login = Carol()

staging = Staging(login)

X_cols =["AveRooms", "HouseAge", "Latitude", "MedInc", "AveOccup", "Longitude", "Population", "AveBedrms"]
y_cols = ["target"]
roi_cols = X_cols + y_cols

data = staging.fetch_parquet(staging_name="samples",
                             connector_name="california_house",
                             cds=True,
                             columns=roi_cols)

#########################
# ============== Model
#########################

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

X_train, X_test, y_train, y_test = train_test_split(
    data[X_cols],
    data[y_cols],
    test_size=0.20,
    random_state=42
)

mlp_model = MLPRegressor(random_state=42, max_iter=500)
mlp_model.fit(X_train, y_train["target"].values)
y_pred = mlp_model.predict(X_test)

#########################
# ============== Evaluate Model
#########################

import numpy as np

y_real = list(y_test["target"].values)
residual = list(y_test["target"].values) - y_pred

mse_f = np.mean(residual**2)
mae_f = np.mean(abs(residual))
rmse_f = np.sqrt(mse_f)

print(f"Mean Squared Error (MSE): {mse_f}.")
print(f"Mean Absolute Error (MAE): {mae_f}.")
print(f"Root Mean Squared Error (RMSE): {rmse_f}.")

#########################
# ============== Storage
#########################

from pycarol import Storage
storage = Storage(login)
storage.save("chp_mlp_regressor", mlp_model, format="pickle")
