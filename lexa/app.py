import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Define the required scopes
SCOPES = ['https://www.googleapis.com/auth/generative-language']

# Load your service account credentials
service_account_info = st.secrets["service_account"]
credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

# Refresh the token if necessary
credentials.refresh(Request())

# Use the credentials to authenticate your API requests
genai.configure(credentials=credentials)

# Set Streamlit page configuration
st.set_page_config("Lexa Multi PDF Agent", page_icon=":scroll:")

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(credentials=credentials, model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(credentials=credentials, model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def clear_chat_history():
    # Delete the index file
    if os.path.isfile("faiss_index/index.faiss"):
        os.remove("faiss_index/index.faiss")

        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs again and ask me a question from the uploaded file."}
        ]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(credentials=credentials, model="models/embedding-001")

    if not os.path.isfile("faiss_index/index.faiss"):
        st.error("No index file found. Please upload some PDFs and try again.")
        return None
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to load the database or process query: {e}")
        return None
    
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def extract_all_text(pdf_docs):
    text = get_pdf_text(pdf_docs)
    st.text_area("Extracted Text", text, height=300)
    return text

def search_within_pdf(pdf_docs, search_term):
    text = get_pdf_text(pdf_docs)
    if search_term.lower() in text.lower():
        st.success(f"'{search_term}' found in the document.")
    else:
        st.error(f"'{search_term}' not found in the document.")

def download_text(text, filename="extracted_text.txt"):
    st.download_button(
        label="Download Extracted Text",
        data=text,
        file_name=filename,
        mime="text/plain"
    )

# Add padding for footer:
st.write("<style>body { padding-bottom: 80px; }</style>", unsafe_allow_html=True)

# Initialize session state:
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask me a question!"}]

def main():
    st.header("📚 Lexa - A Multi PDF Agent 🤖")
    
    with st.sidebar:
        st.image("lexa/Robot.jpg")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ",
                                    accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("No PDF files uploaded. Please upload a file and try again.")
            else:
                with st.spinner("Processing..."):  # user friendly message.
                    raw_text = get_pdf_text(pdf_docs)  # get the pdf text
                    text_chunks = get_text_chunks(raw_text)  # get the text chunks
                    get_vector_store(text_chunks)  # create vector store
                    st.success("Done")

        if st.button("Extract All Text"):
            if not pdf_docs:
                st.error("No PDF files uploaded. Please upload a file and try again.")
            else:
                extracted_text = extract_all_text(pdf_docs)
                download_text(extracted_text)

        search_term = st.text_input("Search within PDF")
        if st.button("Search"):
            if not pdf_docs:
                st.error("No PDF files uploaded. Please upload a file and try again.")
            elif not search_term:
                st.error("Please enter a search term.")
            else:
                search_within_pdf(pdf_docs, search_term)

        st.button("Clear Chat History", on_click=clear_chat_history)

        st.write("---")
        st.title("💡 Predefined Questions")
        predefined_questions = [
            "What is the main topic of the document?",
            "Can you summarize the document?",
            "What are the key points discussed?",
            "Who is the author of the document?"
        ]
        
        for question in predefined_questions:
            if st.button(question):
                st.session_state.messages.append({"role": "user", "content": question})
                response = user_input(question)
                if response is not None:
                    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                else:
                    st.error("No response received.")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs again and ask me a question from the uploaded file."}
        ]    

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.write(message["content"])
            else:
                st.error("Message content not found")

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner(f"Lexa is thinking..."):
                response = user_input(prompt)
                if response is not None:
                    placeholder = st.empty()
                    full_response = ''
                    try:
                        for item in response["output_text"]:
                            full_response += item
                            placeholder.markdown(full_response)
                    except (TypeError, KeyError) as e:
                        st.error("An error occurred while processing the response.")
                    else:
                        placeholder.markdown(full_response)  
                else:
                    st.error("No response received.")
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)                  

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/DeeptiOP" target="_blank">Deepti's Developer App</a> | Made with ❤️ 🤖
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
