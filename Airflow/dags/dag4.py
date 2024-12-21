from datetime import datetime, timedelta
from airflow import DAG
from dotenv import load_dotenv
from airflow.operators.python import PythonOperator
import pandas as pd
from io import BytesIO
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import numpy as np
import os
import requests
import io
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import requests
import numpy as np
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
import openai
from typing import Any, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
import boto3
import re
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import numpy as np
from airflow.hooks.base_hook import BaseHook
from sqlalchemy.exc import SQLAlchemyError
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv



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


dag = DAG('task4',
          default_args=default_args,
          description='A DAG to convert csv to daaframe',
          schedule_interval=None)

def process_recent_csv_from_s3(bucket_name, **kwargs):
 
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,  # Replace with your actual key
            aws_secret_access_key=aws_secret_access_key,  # Replace with your actual secret key
           # region_name='us-east-1'  # Ensure this is the correct region of your S3 bucket
        )

        # File name in the bucket
        file_key = 'status1.csv'

        # Get the object from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        obj_content = obj['Body'].read()

        # Read the content of the CSV into a pandas DataFrame
        df = pd.read_csv(BytesIO(obj_content))
        df = df.dropna(subset=['diagnosis', 'symptoms'])  # Drop rows with NaN
        df = df[
            df['diagnosis'].str.strip() != ''  # Drop rows with empty 'Extracted Text'
        ]
        df = df[
            df['symptoms'].str.strip() != ''  # Drop rows with empty 'Response'
        ]
        df = df.drop_duplicates()
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        # Display the first few rows of the dataframe
        csv_data = df.to_csv(index=False)
        return csv_data

    


from pinecone import Pinecone

def initialize_pinecone():
    try:
        # Load Pinecone configuration
        api_key = ''

        # Create Pinecone instance
        pc = Pinecone(api_key=api_key)
        print("Pinecone initialized successfully.")
        return pc
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        raise

def generate_question_embedding(question_text, embed_model="text-embedding-3-small"):
            # Generate embedding for the question using the specified model
            response = openai.Embedding.create(input=question_text, engine=embed_model)
            embedding = response.data[0].embedding
            if isinstance(embedding, list):
                # If embedding is already a list, no need to convert
                return embedding
            elif isinstance(embedding, np.ndarray):
                # If embedding is a NumPy array, convert it to a list
                return embedding.tolist()
            else:
                # Handle other types of embeddings if necessary
                raise ValueError("Unsupported embedding type: {}".format(type(embedding)))

from pinecone import ServerlessSpec, Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore

from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid

import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone


def cc_pinecone(dataframe):
    load_dotenv()  # Load environment variables from .env file

    # Load OpenAI API key and Pinecone API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    api_key = ''
    environment = 'us-east1-gcp'

    # Initialize OpenAI embeddings and Pinecone client
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=api_key, environment=environment)
    
    index_name = "chatbot2"
    question_namespace = "symptoms"
    answer_namespace = "diagnosis"


    index = pc.Index(name=index_name)

    # Iterate through the DataFrame rows
    for idx, row in dataframe.iterrows():
        # Ensure valid text inputs
        question_text = row.get('symptoms', "").strip()
        answer_text = row.get('diagnosis', "").strip()

        if not question_text:
            print(f"Skipping invalid question at row {idx}")
            continue

        if not answer_text:
            print(f"Skipping invalid answer at row {idx}")
            continue

        # Unique IDs to avoid overwriting vectors
        question_id = f"question_{uuid.uuid4()}"
        answer_id = f"answer_{uuid.uuid4()}"

        # Generate embeddings for the question and answer
        try:
            question_embedding = embeddings_model.embed_query(question_text)
            answer_embedding = embeddings_model.embed_query(answer_text)

            # Log the embeddings
            print(f"Generated question embedding for row {idx}: {question_embedding[:5]}...")  # Print a slice for brevity
            print(f"Generated answer embedding for row {idx}: {answer_embedding[:5]}...")
        except Exception as e:
            print(f"Error generating embeddings for row {idx}: {e}")
            continue

        # Prepare vectors with metadata
        question_vector = {
            "id": question_id,
            "values": question_embedding,
            "metadata": {"text": question_text}
        }
        answer_vector = {
            "id": answer_id,
            "values": answer_embedding,
            "metadata": {"text": answer_text}
        }

        # Log the payload
        print(f"Upserting question vector: {question_vector}")
        print(f"Upserting answer vector: {answer_vector}")

        # Upsert vectors into Pinecone with appropriate namespaces
        try:
            index.upsert(vectors=[question_vector], namespace=question_namespace)
            index.upsert(vectors=[answer_vector], namespace=answer_namespace)
        except Exception as e:
            print(f"Error upserting vectors for row {idx}: {e}")
            continue

    print("Data successfully upserted into Pinecone.")





def ingest_pinecone(task_instance, **kwargs):
    ti = kwargs['ti']
    clean_csv_data = task_instance.xcom_pull(task_ids='process_recent_csv_from_s3_task')
    dataframe = pd.read_csv(io.StringIO(clean_csv_data))
    print("First five rows of the cleaned DataFrame:")
    dataframe.reset_index(inplace=True)
    dataframe.rename(columns={'index': 'id'}, inplace=True)
    dataframe.loc[:, 'id'] += 1
    print(dataframe.head())
    initialize_pinecone()
    cc_pinecone(dataframe)





with dag:
    process_csv = PythonOperator(
        task_id='process_recent_csv_from_s3_task',
        python_callable=process_recent_csv_from_s3,
        op_kwargs={'bucket_name': 'healtcareapplications'},
    )

    ingestpinecone = PythonOperator(
    task_id='ingestpinecone_task',
    python_callable=ingest_pinecone,
    provide_context=True
)
   
  
    


    process_csv >> ingestpinecone 
