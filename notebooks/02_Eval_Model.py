# Databricks notebook source
# MAGIC %pip uninstall -y mlflow-skinny
# MAGIC %pip install git+https://github.com/mlflow/mlflow.git@mlflow-3 catboost
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("eval_table_name", "", "Eval Data Table")
dbutils.widgets.text("model_name", "", "Model Name")
dbutils.widgets.text("model_version", "", "Model Version")
dbutils.widgets.text("experiment_name", "", "Experiment Name")
                     


# COMMAND ----------


eval_table_name = dbutils.widgets.get("eval_table_name")
target = dbutils.widgets.get("target")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

# COMMAND ----------

user = user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

EXP_NAME = f"/Users/{user}/{experiment_name}"

if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME)
mlflow.set_experiment(EXP_NAME)

# COMMAND ----------

# eval_dataset = mlflow.data.load_delta(table_name=eval_table_name)
# spark_eval_df = eval_dataset.df
# df = spark_eval_df.toPandas().dropna()

df = spark.table(eval_table_name).toPandas().dropna()



# COMMAND ----------

import mlflow



model_type = "classifier"

model_uri = "models:/" + model_name + "/" + model_version 
# can also fetch model ID and use that for URI instead as described below

with mlflow.start_run(run_name="evaluation") as run:
  mlflow.evaluate(
    model=model_uri,
    data=df,
    targets=target,
    model_type=model_type
  )

# COMMAND ----------

print(model_uri)
