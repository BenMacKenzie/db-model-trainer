# Databricks notebook source
# MAGIC %pip install git+https://github.com/mlflow/mlflow@mlflow-3 catboost
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %pip install catboost
# MAGIC

# COMMAND ----------

dbutils.widgets.text("target", "", "Target")
dbutils.widgets.text("training_table_name", "", "Training Data Table")
dbutils.widgets.text("eval_table_name", "", "Eval Data Table")
dbutils.widgets.text("experiment_name", "", "Expirement Name")


# COMMAND ----------

import catboost
from catboost import *
from mlflow.models import infer_signature
import mlflow



# COMMAND ----------

training_table_name = dbutils.widgets.get("training_table_name")
eval_table_name = dbutils.widgets.get("eval_table_name")
target = dbutils.widgets.get("target")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------

user = user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

#EXP_NAME = f"/Users/ben.mackenzie@databricks.com/{experiment_name}"
EXP_NAME = f"{user}/{experiment_name}"

if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME)
mlflow.set_experiment(EXP_NAME)

# COMMAND ----------

spark_df = spark.table(training_table_name)
df = spark_df.toPandas().dropna()
y = df.pop(target)
X = df





# COMMAND ----------

# MAGIC %md
# MAGIC #### Find Categorical columns

# COMMAND ----------

cat_features_by_type = [col for col in df.columns if df.dtypes[col] == 'object']

N = 10
cat_features_by_distribution = []
for col in df.columns:
    if col in cat_features_by_type:
      continue
    if spark_df.select(col).distinct().count() <= N:
        cat_features_by_distribution.append(col)

cat_features = list(set(cat_features_by_type + cat_features_by_distribution))


# COMMAND ----------

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'cat_features': cat_features,
    'early_stopping_rounds': 10,
    'random_seed': 42,
    'verbose': False,
    'train_dir': '/tmp/catboost_info'
}

cv_dataset = Pool(X, y, cat_features=cat_features)
df_sample = X.head(2)
signature = infer_signature(X, y)

scores = cv(
        cv_dataset,
        params,
        fold_count=5,
        seed=42,
        plot=False
    )
best_iteration = scores['test-AUC-mean'].idxmax()
params["iterations"] = best_iteration
model = CatBoostClassifier(**params)
model.fit(X, y, cat_features=cat_features)

# Start MLflow run
with mlflow.start_run() as run:
    metrics = scores.iloc[best_iteration].to_dict()
    mlflow.log_metrics(metrics)
    mlflow.log_params(params)
    model_info = mlflow.catboost.log_model(model, 'model', signature=signature, input_example=df_sample)

