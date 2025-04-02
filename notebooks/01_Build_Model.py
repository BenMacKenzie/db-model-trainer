# Databricks notebook source
dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("table_name", "", "Training Data Table")
dbutils.widgets.text("expirement_name", "", "Expirement Name")


# COMMAND ----------

pip install catboost mlflow mlflow-skinny[databricks]>=2.4.1 


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import catboost
from catboost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from mlflow.models import infer_signature
import mlflow



# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

table_name = dbutils.widgets.get("table_name")
df = spark.table(table_name)
target = dbutils.widgets.get("target")
experiment_name = dbutils.widgets.get("expirement_name")

# COMMAND ----------

df= df.toPandas().dropna()
target = df.pop(target)
X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8)

# COMMAND ----------

categories = ['Cabin', 'Pclass', 'Sex', 'Embarked', 'Ticket', 'PassengerId', 'Name']
titanic_train_pool = Pool(X_train, y_train, cat_features=categories)
titanic_test_pool = Pool(X_test, y_test, cat_features=categories)

# COMMAND ----------

experiment_name = f"/Users/ben.mackenzie@databricks.com/experiments/{experiment_name}"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# # Check if the experiment exists
# experiment = mlflow.get_experiment_by_name(experiment_name)

# if experiment is None:
#     # Create the experiment if it does not exist
#     experiment_id = mlflow.create_experiment(experiment_name)
# else:
#     experiment_id = experiment.experiment_id

# # Set the experiment
# mlflow.set_experiment(experiment_name)

# COMMAND ----------

experiment

# COMMAND ----------

df_sample = X_train.head(2)
signature = infer_signature(X_train, y_train)


with mlflow.start_run() as run:
    
    model = CatBoostClassifier(iterations=100, depth=2, learning_rate=1, loss_function='Logloss', allow_writing_files=False)
    model.fit(titanic_train_pool, eval_set=titanic_test_pool, early_stopping_rounds=20, plot=False)

    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

  
    # Log model, metrics, and parameters
    mlflow.catboost.log_model(model, 'model', signature=signature, input_example=df_sample)
  
    mlflow.log_metrics({
        'test_accuracy': accuracy,
        'test_roc_auc': auc
    })

    mlflow.log_params(model.get_all_params())

    print(f"Logged to MLflow with run ID: {run.info.run_id}")


# COMMAND ----------

import mlflow

logged_model = f"runs:/{run.info.run_id}/model"


# COMMAND ----------


catalog = "bmac"
schema = "default"
model_name = "titanic"

#mlflow.register_model(logged_model, f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

#model = mlflow.catboost.load_model(f"models:/{catalog}.{schema}.{model_name}@champion")
