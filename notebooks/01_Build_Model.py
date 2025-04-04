# Databricks notebook source
dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("table_name", "", "Training Data Table")
dbutils.widgets.text("experiment_name", "", "Expirement Name")


# COMMAND ----------

pip install catboost mlflow


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

table_name = dbutils.widgets.get("table_name")
target = dbutils.widgets.get("target")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------

EXP_NAME = f"/Users/ben.mackenzie@databricks.com/{experiment_name}"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME)
mlflow.set_experiment(EXP_NAME)

# COMMAND ----------

df = spark.table(table_name)
df= df.toPandas().dropna()
target = df.pop(target)
X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8)

# COMMAND ----------

categories = ['Cabin', 'Pclass', 'Sex', 'Embarked', 'Ticket', 'PassengerId', 'Name']
titanic_train_pool = Pool(X_train, y_train, cat_features=categories)
titanic_test_pool = Pool(X_test, y_test, cat_features=categories)

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
