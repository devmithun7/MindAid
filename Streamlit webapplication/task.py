from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from pinecone import Pinecone

import streamlit as st

from transformers import pipeline

load_dotenv()

# Configure OpenAI API key
OPENAI_API = os.getenv("OPENAI_API_KEY")



emotion_model = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
                        

def run_llm1(query: str) -> str:
        information = query
        summary_template = """
        Analyze the following text and determine if it indicates something negative, such as distress, discomfort, or an adverse mental or emotional state.
        Provide a clear and definitive answer: "Yes" or "No"

        Text: \"{information}\"
"""

        summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template
        )

        openai_api_key= os.environ['OPENAI_API_KEY']
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

        res = chain.invoke(input={"information": information})
            # Update the session state variable instead of a local variable
        if res['text'] == 'No':
            write = "The text is not a symptom."
            return write
        else:
                summary_template = """
            # Extract Key Phrases
       
        Extract key phrases from the following text that describe emotions, symptoms, or situations. 
        Return the results as comma-separated values.

        Text: "{information}"
        """
                summary_prompt_template = PromptTemplate(
                input_variables=["information"], template=summary_template
            )

                openai_api_key= os.environ['OPENAI_API_KEY']
                llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



                chain = LLMChain(llm=llm, prompt=summary_prompt_template)

                keywords = chain.invoke(input={"information": information})
                pc = Pinecone(
                api_key= os.getenv("PINECONE_API_KEY"),
                environment="us-east1-gcp"
                    )
                index_name= 'chatbot'
                index = pc.Index(index_name)

                embeddings = OpenAIEmbeddings()
                set_b_only_questions_vectors = embeddings.embed_query(query)
                question_t3 = index.query(vector=set_b_only_questions_vectors, top_k=1, namespace='questions', include_metadata=True)
                symptomsarray=[]
                symptomsarray.append(index.query(vector=set_b_only_questions_vectors, top_k=1, namespace='questions', include_metadata=True, include_values=True))
                if symptomsarray[0].matches[0].score>0.75:
                        answer_ids = []
                        for question in symptomsarray[0].matches: 
                            answer_ids.append(index.query(vector=question.values, top_k=1, namespace='answers', include_metadata=True))
                        information = answer_ids[0].matches[0].metadata['text']
                        summary_template = """
                        You are a chatbot assisting individuals who are struggling with depression on an emergency website.
        Based on the input given by the user, the response is: {information}.
        Craft a supportive and empathetic message to the user using the response, make it elobrate upto 5 points. Acknowledge their feelings, and gently encourage them to visit a doctor or mental health professional if required. Ensure the response is user-friendly, compassionate, and reassuring.
        Give the output in in bulletins points
                        """

                        summary_prompt_template = PromptTemplate(
                        input_variables=["information"], template=summary_template
                    )
                        openai_api_key= os.environ['OPENAI_API_KEY']
                        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



                        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

                        result = chain.invoke(input={"information": information})
                        write = result['text']
                        return write
                else:
                    t = 'no medication was found in our dataset'
                    return t

def run_llm2(query: str) -> str:
        information = query
        summary_template = """
        Is the following text related to healthcare? \"{information}\" Provide a simple yes or no answer.
        """

        summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template
        )

        openai_api_key= os.environ['OPENAI_API_KEY']
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

        res = chain.invoke(input={"information": information})
            # Update the session state variable instead of a local variable
        if res['text'] == 'No':
            write = "The text is not a symptom."
            return write
        else:
                summary_template = """
            extract the symptoms from {information} and just mention the symptoms. 
            """
                summary_prompt_template = PromptTemplate(
                input_variables=["information"], template=summary_template
            )

                openai_api_key= os.environ['OPENAI_API_KEY']
                llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



                chain = LLMChain(llm=llm, prompt=summary_prompt_template)

                keywords = chain.invoke(input={"information": information})
                pc = Pinecone(
                api_key= os.getenv("PINECONE_API_KEY"),
                environment="us-east1-gcp"
                    )
                index_name= 'drugsymptom'
                index = pc.Index(index_name)

                embeddings = OpenAIEmbeddings()
                set_b_only_questions_vectors = embeddings.embed_query(query)
                question_t3 = index.query(vector=set_b_only_questions_vectors, top_k=1, namespace='symptom', include_metadata=True)
                symptomsarray=[]
                symptomsarray.append(index.query(vector=set_b_only_questions_vectors, top_k=1, namespace='symptom', include_metadata=True, include_values=True))
                if symptomsarray[0].matches[0].score>0.75:
                        answer_ids = []
                        for question in symptomsarray[0].matches: 
                            answer_ids.append(index.query(vector=question.values, top_k=1, namespace='drug', include_metadata=True))
                        information = answer_ids[0].matches[0].metadata['text']
                        summary_template = """
                        You are a chatbot in a hospital providing the drug name for the patients.
                        For the symptoms the patient has provided this is the drug they should take: {information}.
                        Provide the output in a user friendly manner, in bulletins points
                        """

                        summary_prompt_template = PromptTemplate(
                        input_variables=["information"], template=summary_template
                    )
                        openai_api_key= os.environ['OPENAI_API_KEY']
                        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)



                        chain = LLMChain(llm=llm, prompt=summary_prompt_template)

                        result = chain.invoke(input={"information": information})
                        write = result['text']
                        return write
                else:
                    t = 'no medication was found in our dataset'
                    return t



# Page Configuration
st.set_page_config(
    page_title="Mental Health Assistance",
    page_icon="🤔",
    layout="wide"
)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Define navigation function
def navigate_to(page):
    st.session_state.current_page = page



# Home Page
if st.session_state.current_page == 'home':
    st.title("💭 Mental Health Assistance Platform")
    st.markdown("""
    Welcome to the Mental Health Assistance Platform. This application is designed to provide personalized tools and insights for improving mental health.

    ### Features:
    - **Anxiety Test**: Submit your thoughts, and we'll assess whether you may be experiencing anxiety or depression.
    - **Depression Assistance**: Receive insights and suggestions on how to cope with depression based on your input.
    - **Medication Information**: Learn about medications that may help, tailored to your needs.
    - **Note**: This application uses your data to provide responses. By using this application, you agree to share your data.
    Use the sidebar to navigate through the application.
    """)


# Anxiety Test Page
elif st.session_state.current_page == 'anxiety_test':
    st.title("😟 Emotional State Test")
    st.markdown("""
    Share your thoughts or feelings, this will be used to diagnose your Emotional State.\n
    **Note:** Validations have not been implemented. Please ensure you enter only relevant data.
    """)
    
    anxiety_input = st.text_area("Give your input bellow:")
    if st.button("Submit"):
        if anxiety_input:

            # Load pre-trained emotion detection model

            # Analyze emotion for the input context
            emotion = emotion_model(anxiety_input)[0]['label']

            # Output detected emotion
            st.write("Your current emotion state is: ", emotion)
        else:
                st.warning("Please provide some input before submitting.")             



elif st.session_state.current_page == 'depression_assistance':
    st.title("😞 Depression Assistance")
    st.markdown("""
    Share your challenges, and we'll offer suggestions to support you.
    """)

    # Text input for user to describe their challenges
    depression_input = st.text_area("Describe your challenges or how you feel right now bellow:")
    
    # Button to submit the input
    if st.button("Submit"):
        if depression_input:
            try:
                # Pass the user input to the LLM function
                result = run_llm1(depression_input)
                st.success("Thank you for sharing. Based on your input, here's some advice:")
                st.write(f"- {result}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide some input before submitting.")







# Medication Page
elif st.session_state.current_page == 'medication':
    st.title("💊 Medication")
    st.markdown("""
    Share your symptoms, and we'll provide you a relevant medication.
    """)

    medication_input = st.text_area("Describe your symptoms below:")
    if st.button("Submit"):
        if medication_input:
            result = run_llm2(medication_input)
            st.success("Thank you for sharing. Based on your input, here's your medication:")
            st.write(f"- {result}")
        else:
            st.warning("Please provide some input before submitting.")


# Sidebar Navigation
with st.sidebar:
    st.title("Navigate")
    st.button("Home", on_click=lambda: navigate_to('home'))
    st.button("Anxiety Test", on_click=lambda: navigate_to('anxiety_test'))
    st.button("Depression Assistance", on_click=lambda: navigate_to('depression_assistance'))
    st.button("Medication Info", on_click=lambda: navigate_to('medication'))
