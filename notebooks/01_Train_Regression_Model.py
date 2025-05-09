# Databricks notebook source
dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("training_table_name", "", "Training Data Table")
dbutils.widgets.text("eval_table_name", "", "Eval Data Table")
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

training_table_name = dbutils.widgets.get("training_table_name")
eval_table_name = dbutils.widgets.get("eval_table_name")
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



# COMMAND ----------

training_dataset = mlflow.data.load_delta(table_name=training_table_name)
spark_train_df = training_dataset.df
df = spark_train_df.toPandas().dropna()
target_col = df.pop(target)
X_train, y_train = df, target_col



eval_dataset = mlflow.data.load_delta(table_name=eval_table_name)
spark_eval_df = eval_dataset.df
df = spark_eval_df.toPandas().dropna()
target_col = df.pop(target)
X_test, y_test = df, target_col



# COMMAND ----------

# MAGIC %md
# MAGIC #### Find Categorical columns

# COMMAND ----------

cat_features_by_type = [col for col in df.columns if X_train.dtypes[col] == 'object']

N = 10
cat_features_by_distribution = []
for col in df.columns:
    if col in cat_features_by_type:
      continue
    if spark_train_df.select(col).distinct().count() <= N:
        cat_features_by_distribution.append(col)

cat_features = list(set(cat_features_by_type + cat_features_by_distribution))


# COMMAND ----------

#categories = ['Cabin', 'Pclass', 'Sex', 'Embarked', 'Ticket', 'PassengerId', 'Name']
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# COMMAND ----------

import pandas as pd

df_sample = X_train.head(2)
signature = infer_signature(X_train, y_train)


with mlflow.start_run() as run:
    
    model = CatBoostRegressor(allow_writing_files=False)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20, plot=False)
    model_info = mlflow.catboost.log_model(model, 'model', signature=signature, input_example=df_sample)

    
    mlflow.log_params(model.get_all_params())
    mlflow.log_input(training_dataset, context="training")
    mlflow.set_tags({"training": training_table_name, "evaluation": eval_table_name})

    print(f"Logged to MLflow with run ID: {run.info.run_id}")

with mlflow.start_run(run_id=run.info.run_id):
    eval_data = pd.concat([X_test, y_test], axis=1)

    results = mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets=target,
        model_type="regressor",
        evaluators="default"
    )
    mlflow.log_input(eval_dataset, context="evaluation")
    print(f"Logged to MLflow with run ID: {run.info.run_id}")

