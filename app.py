import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.write("Ask questions about Zyro Dynamics HR policies and employee information.")


# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

LLM_MODEL = "openai/gpt-oss-20b"

CORPUS_PATH = "./zyro-dynamics-hr-corpus/"


# ---------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------

@st.cache_resource
def load_documents():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)

    documents = loader.load()

    return documents


# ---------------------------------------------------------
# SPLIT DOCUMENTS
# ---------------------------------------------------------

@st.cache_resource
def create_chunks():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------

@st.cache_resource
def create_embeddings():

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings_model


# ---------------------------------------------------------
# VECTOR DATABASE
# ---------------------------------------------------------

@st.cache_resource
def create_vectorstore():

    chunks = create_chunks()

    embeddings_model = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings_model
    )

    return vectorstore


# ---------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------

@st.cache_resource
def create_retriever():

    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

@st.cache_resource
def create_llm():

    llm_model = ChatGroq(
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=500,
        groq_api_key=GROQ_API_KEY
    )

    return llm_model


# ---------------------------------------------------------
# RAG PROMPT
# ---------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are an AI HR Assistant for Zyro Dynamics.

Your task is to answer the user's question using the information
provided in the context.

If the required information is present in the context,
give a clear and accurate answer.

If the information is not present in the context, say:

"I don't have enough information to answer this question."

Do not create, assume, or guess any information.

Context:
{context}

Question:
{question}
"""
)


# ---------------------------------------------------------
# SCOPE / GUARDRAIL PROMPT
# ---------------------------------------------------------

SCOPE_PROMPT = ChatPromptTemplate.from_template(
"""
You are a scope classifier for the Zyro Dynamics HR Assistant.

Decide whether the user's question is related to internal company HR
topics, such as leave policies, attendance, reimbursement,
employee benefits, code of conduct, performance reviews,
onboarding, separation, travel and expense policies,
workplace policies, or other Zyro Dynamics HR topics.

Questions about general knowledge, coding help, programming,
technical questions, entertainment, or unrelated topics are
considered outside the scope.

Respond with exactly one word:

IN_SCOPE or OUT_OF_SCOPE

Question:
{question}
"""
)


REFUSAL_MESSAGE = (
    "I am the Zyro Dynamics HR Assistant and can only help with "
    "questions related to company HR policies and employee information. "
    "I don't have information to answer that question."
)


# ---------------------------------------------------------
# FORMAT DOCUMENTS
# ---------------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ---------------------------------------------------------
# ASK BOT
# ---------------------------------------------------------

def ask_bot(question):

    retriever = create_retriever()

    llm_model = create_llm()

    # Guardrail
    guardrail_chain = (
        SCOPE_PROMPT
        | llm_model
        | StrOutputParser()
    )

    verdict = guardrail_chain.invoke(
        {"question": question}
    ).strip().upper()

    if "OUT_OF_SCOPE" in verdict:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    # Retrieval
    docs = retriever.invoke(question)

    context = format_docs(docs)

    # RAG
    answer_chain = (
        RAG_PROMPT
        | llm_model
        | StrOutputParser()
    )

    response = answer_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return {
        "answer": response,
        "sources": docs
    }


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

question = st.text_input(
    "Ask your HR question:",
    placeholder="Example: What is the company's leave policy?"
)


if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                result = ask_bot(question)

                st.subheader("Answer")

                st.write(result["answer"])

                if result["sources"]:

                    st.subheader("Sources")

                    source_names = []

                    for doc in result["sources"]:

                        source = doc.metadata.get("source")

                        if source:
                            source_names.append(
                                os.path.basename(source)
                            )

                    for source in dict.fromkeys(source_names):

                        st.write(f"📄 {source}")

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )
