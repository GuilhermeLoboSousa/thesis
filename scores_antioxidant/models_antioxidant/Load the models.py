# Databricks notebook source
import joblib

# Try loading the model using joblib
try:
    model = joblib.load('best_classifier_antioxidant_data_tabular_shallow_sgd.h5')
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")


# COMMAND ----------

from tensorflow.keras.models import load_model

model = load_model('best_antioxidant_data_tabular_autoencoder_adam.h5')

# Print the model summary (architecture, layers, etc.)
model.summary()

optimizer = model.optimizer.get_config()
print("Optimizer hyperparameters:", optimizer)

# # Access the layers and their hyperparameters
for layer in model.layers:
    print(layer.name, layer.get_config())

# COMMAND ----------

from tensorflow.keras.models import load_model

model = load_model('best_antioxidant_data_tabular_autoencoder_rmsprop.h5')

# Print the model summary (architecture, layers, etc.)
model.summary()

optimizer = model.optimizer.get_config()
print("Optimizer hyperparameters:", optimizer)

# # Access the layers and their hyperparameters
for layer in model.layers:
    print(layer.name, layer.get_config())

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_knn.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_gboosting.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_linear_svc.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_lr.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_nn.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_rf.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

a="best_classifier_antioxidant_data_tabular_shallow_svc.h5"
import joblib

# Try loading the model using joblib
try:
    model = joblib.load(a)
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

from tensorflow.keras.models import load_model

model = load_model('best_rnn_model_antioxidant_esm.h5')

# Print the model summary (architecture, layers, etc.)
model.summary()

optimizer = model.optimizer.get_config()
print("Optimizer hyperparameters:", optimizer)

# # Access the layers and their hyperparameters
for layer in model.layers:
    print(layer.name, layer.get_config())

# COMMAND ----------

from tensorflow.keras.models import load_model

model = load_model('best_rnn_model_antioxidant_nlf.h5')

# Print the model summary (architecture, layers, etc.)
model.summary()

optimizer = model.optimizer.get_config()
print("Optimizer hyperparameters:", optimizer)

# # Access the layers and their hyperparameters
for layer in model.layers:
    print(layer.name, layer.get_config())

