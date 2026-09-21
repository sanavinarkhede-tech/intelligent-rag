import streamlit as st
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.write("Ask questions about Zyro Dynamics HR policies.")


# --------------------------------------------------
# API KEY
# --------------------------------------------------

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# --------------------------------------------------
# CORPUS PATH
# --------------------------------------------------

CORPUS_PATH = "./zyro-dynamics-hr-corpus"


if not os.path.exists(CORPUS_PATH):
    st.error(f"HR corpus folder not found: {CORPUS_PATH}")
    st.stop()


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

@st.cache_resource
def load_documents():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)

    documents = loader.load()

    return documents


with st.spinner("Loading HR documents..."):
    documents = load_documents()


st.success(f"Loaded {len(documents)} HR documents")


# --------------------------------------------------
# SPLIT DOCUMENTS
# --------------------------------------------------

@st.cache_resource
def create_chunks(_documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(_documents)

    return chunks


with st.spinner("Creating document chunks..."):
    chunks = create_chunks(documents)


st.write(f"Created {len(chunks)} document chunks")


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

@st.cache_resource
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


with st.spinner("Loading embedding model..."):
    embeddings_model = create_embeddings()


# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def create_vectorstore(_chunks, _embeddings):

    vectorstore = FAISS.from_documents(
        _chunks,
        _embeddings
    )

    return vectorstore


with st.spinner("Creating FAISS vector database..."):
    vectorstore = create_vectorstore(
        chunks,
        embeddings_model
    )


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        max_tokens=500
    )

    return llm


llm_model = create_llm()


# --------------------------------------------------
# RAG PROMPT
# --------------------------------------------------

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


# --------------------------------------------------
# SCOPE PROMPT
# --------------------------------------------------

SCOPE_PROMPT = ChatPromptTemplate.from_template(
"""
You are a scope classifier for the Zyro Dynamics HR Assistant.

Decide whether the user's question is related to internal company HR topics,
such as leave policies, attendance, reimbursement, employee benefits,
code of conduct, performance reviews, onboarding, separation, travel
and expense policies, workplace policies, or other Zyro Dynamics HR topics.

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


# --------------------------------------------------
# ASK BOT
# --------------------------------------------------

def ask_bot(question):

    # Scope check
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


    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # Generate answer
    chain = (
        RAG_PROMPT
        | llm_model
        | StrOutputParser()
    )

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


    return {
        "answer": response,
        "sources": docs
    }


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

question = st.text_input(
    "Ask your HR question:",
    placeholder="Example: What is the company's leave policy?"
)


if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            result = ask_bot(question)


        st.subheader("Answer")

        st.write(result["answer"])


        if result["sources"]:

            st.subheader("Sources")

            shown_sources = set()

            for doc in result["sources"]:

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                if source not in shown_sources:

                    st.write(f"📄 {source}")

                    shown_sources.add(source)
