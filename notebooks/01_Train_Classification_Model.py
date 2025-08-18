# Databricks notebook source
dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("training_table_name", "", "Training Data Table")
dbutils.widgets.text("eval_table_name", "", "Eval Data Table")
dbutils.widgets.text("experiment_name", "", "Expirement Name")


# COMMAND ----------

# MAGIC
# MAGIC %pip install git+https://github.com/mlflow/mlflow@mlflow-3 catboost
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

import catboost
from catboost import *
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import mlflow



# COMMAND ----------

training_table_name = dbutils.widgets.get("training_table_name")
eval_table_name = dbutils.widgets.get("eval_table_name")
target = dbutils.widgets.get("target")
experiment_name = dbutils.widgets.get("experiment_name")


# COMMAND ----------

user = user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

EXP_NAME = f"/Users/{user}/{experiment_name}"

if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME)
mlflow.set_experiment(EXP_NAME)

# COMMAND ----------

# training_dataset = mlflow.data.load_delta(table_name=training_table_name)
# spark_train_df = training_dataset.df
# df = spark_train_df.toPandas().dropna()
# target_col = df.pop(target)
# X_train, y_train = df, target_col



# eval_dataset = mlflow.data.load_delta(table_name=eval_table_name)
# spark_eval_df = eval_dataset.df
# df = spark_eval_df.toPandas().dropna()
# target_col = df.pop(target)
# X_test, y_test = df, target_col


spark_df = spark.table(training_table_name)
df = spark_df.toPandas().dropna()
target_df = df.pop(target)
X_train, X_test, y_train, y_test = train_test_split(df, target_df, test_size=0.2, random_state=42)




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
    if spark_df.select(col).distinct().count() <= N:
        cat_features_by_distribution.append(col)

cat_features = list(set(cat_features_by_type + cat_features_by_distribution))


# COMMAND ----------


train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# COMMAND ----------

import pandas as pd

df_sample = X_train.head(2)
signature = infer_signature(X_train, y_train)


model = CatBoostClassifier(iterations=100, depth=2, learning_rate=1, loss_function='Logloss', allow_writing_files=False)

#use cv here rather than eval_set.
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20, plot=False)

with mlflow.start_run() as run:
    
    model_info = mlflow.catboost.log_model(model, 'model', signature=signature, input_example=df_sample)
    eval_data = pd.concat([X_test, y_test], axis=1)
    results = mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets=target,
        model_type="classifier",
        evaluators="default"
    )

    mlflow.log_params(model.get_all_params())
    mlflow.set_tags({"training": training_table_name})

    print(f"Logged to MLflow with run ID: {run.info.run_id}")



