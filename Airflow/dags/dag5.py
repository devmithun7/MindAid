from datetime import datetime
from airflow import DAG, task
from dotenv import load_dotenv
from airflow.operators.python import PythonOperator
import pandas as pd
from io import BytesIO
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import io
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import numpy as np
import os
import openai
from typing import Any, List
import boto3
import re
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import numpy as np
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import io
import re



openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 3),
    'retries': 1,
}

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
bucket_name = os.getenv("S3_BUCKET_NAME")

session = boto3.Session(
    aws_access_key_id= aws_access_key_id, #kept blank due to privacy issue,
    aws_secret_access_key=aws_secret_access_key,#kept blank due to privacy issue,
)

s3 = session.client('s3')


dag = DAG('process_csv_data_dag',
          default_args=default_args,
          description='A DAG to convert csv to daaframe',
          schedule_interval=None)

def process_recent_csv_from_s3(bucket_name, **kwargs):
    # Use boto3 to create an S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id='',  # Replace with your actual key
            aws_secret_access_key='',  # Replace with your actual secret key
           # region_name='us-east-1'  # Ensure this is the correct region of your S3 bucket
        )

        # File name in the bucket
        file_key = 'status.csv'

        # Get the object from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        obj_content = obj['Body'].read()

        # Read the content of the CSV into a pandas DataFrame
        df = pd.read_csv(BytesIO(obj_content))

        # Display the first few rows of the dataframe
        csv_data = df.to_csv(index=False)
        return csv_data

    except NoCredentialsError:
        print("Error: AWS credentials are not available.")
    except ClientError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")





def datacleaning(task_instance, **kwargs):
    # Pull CSV data from XCom
    csv_data = task_instance.xcom_pull(task_ids='process_recent_csv_from_s3_task')
    
    # Read CSV data into DataFrame
    dataframe = pd.read_csv(io.StringIO(csv_data))
    
    # Drop the unnamed column (index column) and keep only 'statement' and 'status'
    dataframe = dataframe[['statement', 'status']]

    # Rename columns to 'symptoms' and 'diagnosis'
    dataframe.rename(columns={'statement': 'symptoms', 'status': 'diagnosis'}, inplace=True)
    
    # Drop rows with NaN values
    dataframe.dropna(inplace=True)
    
    # Clean text: Keep alphanumeric characters and spaces
    cleaned_df = dataframe.applymap(
        lambda x: ''.join(c for c in str(x) if c.isalnum() or c.isspace()) 
        if isinstance(x, str) else x
    )
    
    return cleaned_df

import time    

def contains_repetitive_pattern(text, threshold=3):
    """
    Checks if the input text contains repetitive patterns.
    
    Args:
    text (str): The text to analyze.
    threshold (int): The minimum number of repetitions to consider the text as repetitive.
    
    Returns:
    bool: True if repetitive patterns are detected, otherwise False.
    """
    # Find repeated words/phrases
    repeated_patterns = re.findall(r"(\b\w+\b)(?:\s+\1){%d,}" % (threshold - 1), text, re.IGNORECASE)
    return len(repeated_patterns) > 0

def openaiextraction(task_instance, **kwargs):
    ti = kwargs['ti']
    # Pull the cleaned DataFrame directly from XCom
    dataframe = ti.xcom_pull(task_ids='datacleaning_task')

    # Check if the data is already a DataFrame; convert if necessary
    if not isinstance(dataframe, pd.DataFrame):
        dataframe = pd.read_csv(io.StringIO(dataframe))
        
    # Prompt for extracting comma-separated values
    summary_template = """
    Extract key phrases from the following text that describe emotions, symptoms, or situations. 
    Return the results as comma-separated values.

    Text: "{symptoms}"
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["symptoms"], template=summary_template
    )
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    result_df = pd.DataFrame(columns=['diagnosis', 'symptoms'])
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    for index, row in dataframe.iterrows():
        symptoms_data = row['symptoms']

        # Skip rows with repetitive patterns
        if contains_repetitive_pattern(symptoms_data):
            print(f"Row {index + 1}: Repetitive pattern detected, skipping...")
            continue

        # Invoke the LLM chain with the symptoms
        try:
            res = chain.invoke(input={"symptoms": symptoms_data})
            response_text = res.get('text', '').strip()
            
            # Print the output for each row
            print(f"Row {index + 1}: {response_text}")
            
            # Append results to the DataFrame
            new_row = pd.DataFrame({
                'diagnosis': [row['diagnosis']],
                'symptoms': [response_text]
            })
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
        
        # Throttle API calls
        if index % 850 == 0:
            time.sleep(300)
            print("Throttling requests...")

    # Cleaning the DataFrame: Remove empty rows and unwanted terms
    result_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    result_df.dropna(inplace=True)
    
    # Save to CSV and upload to S3
    filename = "status1.csv"
    result_df.to_csv(filename, index=False)
    with open(filename, "rb") as f:
        s3.upload_fileobj(f, S3_BUCKET_NAME, filename)

    # Push processed data to XCom for further tasks
    ti.xcom_push(key='processed_data', value=result_df.to_json(orient='split'))
    
    return result_df












trigger = TriggerDagRunOperator(
    task_id = 'trigger_target_dag',
    trigger_dag_id = 'ingest_to_pinecone1_data_dag',
    wait_for_completion = True,
    execution_date = '{{ ds }}'


)




with dag:
    process_csv = PythonOperator(
        task_id='process_recent_csv_from_s3_task',
        python_callable=process_recent_csv_from_s3,
        op_kwargs={'bucket_name': 'healtcareapplications'},
    )

    clean_csv = PythonOperator(
    task_id='datacleaning_task',
    python_callable=datacleaning,
    provide_context=True
)
    
    print_first_five_task = PythonOperator(
            task_id='openaiextraction_task',
            python_callable=openaiextraction,
            provide_context=True
        )
  
    


    process_csv >> clean_csv >> print_first_five_task 
 
