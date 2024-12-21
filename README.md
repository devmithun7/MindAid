# Mental Health Counseling Analysis

This repository contains the implementation of a project analyzing mental health counseling transcripts. The primary goals of this project are to profile and clean the dataset, perform exploratory data analysis (EDA), build predictive machine learning models, and create a web application for practical use. Below is an overview of the repository's contents and functionalities.

- Application Link: http://3.88.118.67:8501/
---

## Architecture Diagram

<img src="https://github.com/devmithun7/Legacy-Task/blob/main/architecture_diagram_task.png" alt="Architecture Diagram" width="600"/>

---

## Repository Structure
### 1. **Data Profiling**
- Folder: `data_profiling`
- **Description**: This folder contains scripts and notebooks for profiling and understanding the dataset. Python tools like `pandas` and `y_dataprofiling` have been used for detailed profiling to identify key data characteristics, outliers, and potential data quality issues.

### 2. **Data Processing and EDA**
- **Description**: Comprehensive data cleaning and exploratory analysis have been performed, including:
  - Handling missing values
  - Removing duplicates
  - Standardizing text and numeric fields
  - Visualizing trends and distributions
- **Tools Used**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `ydata_profiling`
- **Files**: The corresponding notebooks for these steps are available in the Airflow and Machine Learning Model folders.

### 3. **Machine Learning Models**

This folder contains two machine learning models designed to derive insights from counseling data:

1. **Classification Model 1**: Predicts mental disorders based on counseling transcripts using NLP techniques.
   - Classes: Bipolar Type-2, Depression, Normal, Bipolar Type-1.
   - Workflow: Preprocessing (tokenization, text cleaning, TF-IDF/embeddings), modeling (Logistic Regression, Random Forest, Neural Networks), and evaluation (accuracy, precision, recall, F1-score).

2. **Classification Model 2**: Predicts mood swings based on supervised learning.
   - Classes: High, Medium, Low.
   - Workflow: Preprocessing (handling missing values, feature encoding/scaling), modeling (Decision Trees, Gradient Boosting, SVM), and evaluation (confusion matrix, accuracy, ROC-AUC).


### 4 **LLM-Based Application - Mental Health Counselor Support**

This application is an LLM-powered solution designed to assist mental health counselors by generating advice tailored to help their patients. The application leverages a fine-tuned version of the **tiiuae_falcon_7b** model to deliver contextually relevant and actionable insights.

- **Free-Text Input**: Accepts detailed descriptions from counselors about their patients' challenges and concerns.
- **Generated Advice**: Produces actionable, empathetic, and insightful guidance on how to address the described mental health issues.
- **Custom Fine-Tuning**: The application uses a fine-tuned version of the **tiiuae_falcon_7b** model, ensuring responses are specifically aligned with mental health support and counseling practices.



### 5. **Web Application - Mental Health Counseling and LLM Integration**

This web application is part of a larger project and focuses on providing personalized mental health assistance by leveraging a pre-trained large language model (GPT-3.5 Turbo) and advanced similarity search techniques. The app combines user-friendly interfaces with powerful backend technologies to deliver relevant information and actionable insights for mental health challenges.


- **Interactive User Interface**: Built with **Streamlit**, enabling users to:
  - Enter free-text descriptions of mental health challenges and receive tailored suggestions via an LLM.
  - Search a database of mental health counseling data to retrieve relevant examples.
  - Assess their current mental health state and receive advice on managing anxiety and depression.
  - Query for medication suggestions based on symptoms.

- **Similarity Search**: Implements **Retrieval-Augmented Generation (RAG)** using **Pinecone** vector databases with namespaces to efficiently retrieve and rank relevant information.

- The application is fully containerized using **Docker**, enabling easy deployment across various platforms. It serves as a practical demonstration of how LLM-powered analysis and retrieval systems can be utilized to address mental health challenges.

---


## Technologies Used
- **Data Profiling**: `pandas_profiling`, `y_dataprofiling`
- **EDA and Data Cleaning**: `Python` (`pandas`, `matplotlib`, `seaborn`), `Airflow`, `NLTK`
- **Machine Learning**: `scikit-learn`, `TensorFlow`/`PyTorch`
- **Databases and Storage**: `Snowflake`, `S3`, `Pinecone`
- **Web Application**: `Streamlit`, `Docker`, `AWS`
- **LLM Integration**: `Open AI`, `tiiuae-falcon-7b`
---


