# Databricks notebook source
# MAGIC %pip install mlflow --upgrade
# MAGIC %restart_python

# COMMAND ----------

# Create widgets to accept model_name, model_version, and job_name
dbutils.widgets.text("model_name", "", "Model Name")
dbutils.widgets.text("model_version", "", "Model Version")
dbutils.widgets.text("job_name", "", "Job Name")

# Retrieve the widget values
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
job_name = dbutils.widgets.get("job_name")

# COMMAND ----------


# REQUIRED: Create notebooks for each task and populate the notebook path here, replacing the INVALID PATHS LISTED BELOW.
# These paths should correspond to where you put the notebooks templated from the example deployment jobs template notebook
# in your Databricks workspace. Choose an evaluation notebook based on if the model is for GenAI or classic ML
evaluation_notebook_path = "/Workspace/ML/mlflow_workshop/deployment_job/evaluation"
approval_notebook_path = "/Workspace/ML/mlflow_workshop/deployment_job/approval"
deployment_notebook_path = "/Workspace/ML/mlflow_workshop/deployment_job/deployment"

# COMMAND ----------

# Create job with necessary configuration to connect to model as deployment job
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
job_settings = jobs.JobSettings(
    name=job_name,
    tasks=[
        jobs.Task(
            task_key="Evaluation",
            notebook_task=jobs.NotebookTask(notebook_path=evaluation_notebook_path),
            max_retries=0,
        ),
        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={"approval_tag_name": "{{task.name}}"}
            ),
            depends_on=[jobs.TaskDependency(task_key="Evaluation")],
            max_retries=0,
        ),
        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(notebook_path=deployment_notebook_path),
            depends_on=[jobs.TaskDependency(task_key="Approval_Check")],
            max_retries=0,
        ),
    ],
    parameters=[
        jobs.JobParameter(name="model_name", default=model_name),
        jobs.JobParameter(name="model_version", default=model_version),
    ],
    queue=jobs.QueueSettings(enabled=True),
    max_concurrent_runs=1,
)

created_job = w.jobs.create(**job_settings.__dict__)
print("Use the job name " + job_name + " to connect the deployment job to the UC model " + model_name + " as indicated in the UC Model UI.")
print("\nFor your reference, the job ID is: " + str(created_job.job_id))
print("\nDocumentation: \nAWS: https://docs.databricks.com/aws/mlflow/deployment-job#connect \nAzure: https://learn.microsoft.com/azure/databricks/mlflow/deployment-job#connect \nGCP: https://docs.databricks.com/gcp/mlflow/deployment-job#connect")

# COMMAND ----------

# Optionally, you can programmatically link the deployment job to a UC model

import mlflow
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

try:
  if client.get_registered_model(model_name):
    client.update_registered_model(model_name, deployment_job_id=created_job.job_id)
except mlflow.exceptions.RestException:
  client.create_registered_model(model_name, deployment_job_id=created_job.job_id)
