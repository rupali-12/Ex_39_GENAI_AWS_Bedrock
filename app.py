import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model to generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Specify your AWS region
region_name = "us-west-2"  # Replace with your desired region

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name=region_name)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # In our testing, Character split works better with this PDF dataset
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ## Create the AI21 Model
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens': 512})
    return llm

def get_llama2_llm():
    ## Create the Meta LLaMA2 Model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_llama2_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
