import streamlit as st # Streamlit helps us to turn data scripts to shareable Web Applications.
import pickle # Saves data in binary files to help us train the model
from dotenv import load_dotenv #Environment variables file takes Open AI API key as input
from streamlit_extras.add_vertical_space import add_vertical_space # To add vertical spaces in streamlit Web App
from PyPDF2 import PdfReader # To read our PDF file input
from langchain.text_splitter import RecursiveCharacterTextSplitter # It helps us to split the large PDF file into smaller chunks
from langchain.embeddings.openai import OpenAIEmbeddings # Langchain Suppurted special Embeddings by Open AI
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.chat_models import ChatOpenAI
import os

with st.sidebar:
    st.title('PDF Chat App')
    st.markdown('''
                This chatapp is made by using:-\n
                * Langchain\n
                * OpenAi Chat models\n
                SUBMITTED BY\n
                Sujit Parida\n
                TO:-\n
                PearlThoughts Hiring Team
                ''')
    add_vertical_space(5)
    st.write('''THANK YOU\n
             Thanks to :-
             https://www.youtube.com/@engineerprompt''')
load_dotenv()
def main():
    st.header('PDF Chat App')
    #Upload a PDF file
    pdf = st.file_uploader("Upload Here(PDF only)", type='pdf')

    #st.write(pdf)

    if pdf is not None:
        
        pdf_reader = PdfReader(pdf)
        

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)

        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl",'rb') as f:
                VectorStore = pickle.load(f)
            st.write('Embeddng is working')    
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl",'wb') as f:            
                pickle.dump(VectorStore,f)
            st.write('Embedding')
        
        query = st.text_input("Ask Question about your PDF: ")
        
        if query:
            docs = VectorStore.similarity_search(query=query,k=3)

            llm = OpenAI(temperature=0,model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            #st.write(docs)

if __name__ == "__main__":
    main()
 