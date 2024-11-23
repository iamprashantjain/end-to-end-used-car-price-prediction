from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger.logging import logging


# Creating an object of the TrainingPipeline class
training_pipeline = TrainingPipeline()

# Configuring the DAG
with DAG(
    "used_car_price_prediction_training_pipeline",
    default_args={"retries": 2},
    description="This is my training pipeline.",
    schedule="@daily",
    start_date=pendulum.datetime(2024, 11, 2, tz="UTC"),  # Corrected typo from start_data to start_date
    catchup=False,
    tags=["machine_learning", "classification", "used_car_price_prediction"],
) as dag:
    
    logging.info("DAG has started successfully")
    
    dag.doc_md = __doc__

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        
        # Calling data ingestion
        train_data_path, test_data_path = training_pipeline.start_data_ingestion()
        
        # Push task to the next component
        ti.xcom_push("data_ingestion_artifact", {"train_data_path": train_data_path, "test_data_path": test_data_path})

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        
        # Pulling data from the previous stage
        data_ingestion_artifact = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifact")
        
        # Performing data transformation
        train_arr, test_arr = training_pipeline.start_data_transformation(data_ingestion_artifact['train_data_path'], data_ingestion_artifact['test_data_path'])
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        
        # Pushing to the next component
        ti.xcom_push("data_transformations_artifact", {'train_arr': train_arr, 'test_arr': test_arr})

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        
        # Pulling data from the previous stage
        data_transformation_artifact = ti.xcom_pull(task_ids='data_transformation', key='data_transformations_artifact')
        train_arr = np.array(data_transformation_artifact['train_arr'])
        test_arr = np.array(data_transformation_artifact['test_arr'])
        
        # Starting model training
        training_pipeline.start_model_training(train_arr, test_arr)

    def push_data_to_s3(**kwargs):
        import os
        bucket_name = os.getenv('BUCKET_NAME')
        artifact_folder = '/app/artifacts'
        os.system(f'aws s3 sync {artifact_folder} s3://{bucket_name}/artifact')

    # Create operators for Airflow
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    
    data_ingestion_task.doc_md = dedent(
        """
        #### Ingestion Task
        This task creates train and test files.
        """
    )

    data_transform_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformations,
    )

    data_transform_task.doc_md = dedent(
        """
        #### Transformation Task
        This task performs data transformations.
        """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )

    model_trainer_task.doc_md = dedent(
        """
        #### Model Trainer Task
        This task performs model training.
        """
    )

    push_data_to_s3_task = PythonOperator(
        task_id="push_data_to_s3",
        python_callable=push_data_to_s3,
    )

    # Define the task flow
    data_ingestion_task >> data_transform_task >> model_trainer_task >> push_data_to_s3_task

    
    #how to run this task in airflow since it cant be run properly in windows
    #we need to write a docker file