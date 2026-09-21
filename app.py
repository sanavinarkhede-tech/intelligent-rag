```python
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.caption("Ask questions related to Zyro Dynamics HR policies and employee information.")


# ============================================================
# GROQ API KEY
# ============================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"].strip()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is empty.")
    st.stop()

if not GROQ_API_KEY.startswith("gsk_"):
    st.error("The Groq API key format looks incorrect.")
    st.stop()


# ============================================================
# LLM CONFIGURATION
# ============================================================

LLM_PROVIDER = "groq"
LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# LOAD HR DOCUMENTS
# ============================================================

CORPUS_PATH = "./zyro-dynamics-hr-corpus"

if not os.path.exists(CORPUS_PATH):
    st.error(
        f"HR corpus folder not found: {CORPUS_PATH}"
    )
    st.info(
        "Make sure the folder 'zyro-dynamics-hr-corpus' "
        "is uploaded to the GitHub repository."
    )
    st.stop()


@st.cache_resource
def load_documents():

    loader = PyPDFDirectoryLoader(CORPUS_PATH)

    documents = loader.load()

    return documents


with st.spinner("Loading HR documents..."):
    documents = load_documents()


if len(documents) == 0:
    st.error("No PDF documents were found in the HR corpus.")
    st.stop()


# ============================================================
# TEXT SPLITTING
# ============================================================

@st.cache_resource
def create_chunks(_documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(_documents)

    return chunks


with st.spinner("Preparing documents..."):
    chunks = create_chunks(documents)


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings():

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings_model


with st.spinner("Loading embedding model..."):
    embeddings_model = create_embeddings()


# ============================================================
# FAISS VECTOR DATABASE
# ============================================================

@st.cache_resource
def create_vectorstore(_chunks, _embeddings_model):

    vectorstore = FAISS.from_documents(
        _chunks,
        _embeddings_model
    )

    return vectorstore


with st.spinner("Creating vector database..."):
    vectorstore = create_vectorstore(
        chunks,
        embeddings_model
    )


# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# GROQ LLM
# ============================================================

@st.cache_resource
def create_llm(api_key):

    llm_model = ChatGroq(
        model=LLM_MODEL,
        temperature=0.7,
        max_tokens=500,
        api_key=api_key
    )

    return llm_model


llm_model = create_llm(GROQ_API_KEY)


# ============================================================
# RAG PROMPT
# ============================================================

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


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# SCOPE / GUARDRAIL PROMPT
# ============================================================

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


# ============================================================
# REFUSAL MESSAGE
# ============================================================

REFUSAL_MESSAGE = (
    "I am the Zyro Dynamics HR Assistant and can only help with "
    "questions related to company HR policies and employee information. "
    "I don't have information to answer that question."
)


# ============================================================
# ASK BOT FUNCTION
# ============================================================

def ask_bot(question: str):

    # ----------------------------
    # Step 1: Scope classification
    # ----------------------------

    guardrail_chain = (
        SCOPE_PROMPT
        | llm_model
        | StrOutputParser()
    )

    verdict = guardrail_chain.invoke(
        {"question": question}
    ).strip().upper()


    # ----------------------------
    # Step 2: Reject unrelated questions
    # ----------------------------

    if "OUT_OF_SCOPE" in verdict:

        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }


    # ----------------------------
    # Step 3: Retrieve documents
    # ----------------------------

    docs = retriever.invoke(question)

    context = format_docs(docs)


    # ----------------------------
    # Step 4: Generate RAG answer
    # ----------------------------

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


    # ----------------------------
    # Step 5: Return answer + sources
    # ----------------------------

    return {
        "answer": response,
        "sources": docs
    }


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        f"Documents loaded: **{len(documents)}**"
    )

    st.write(
        f"Text chunks: **{len(chunks)}**"
    )

    st.write(
        f"LLM: **{LLM_MODEL}**"
    )

    st.divider()

    st.write(
        "This assistant answers questions using "
        "the Zyro Dynamics HR document corpus."
    )


# ============================================================
# USER INPUT
# ============================================================

question = st.text_input(
    "Ask your HR question:",
    placeholder="Example: What is the company's leave policy?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("Ask HR Assistant", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:

                result = ask_bot(question)

                # ----------------------------
                # Display answer
                # ----------------------------

                st.subheader("💬 Answer")

                st.markdown(
                    result["answer"]
                )


                # ----------------------------
                # Display sources
                # ----------------------------

                if result["sources"]:

                    st.subheader("📄 Sources")

                    displayed_sources = set()

                    for doc in result["sources"]:

                        source = doc.metadata.get(
                            "source",
                            "Unknown source"
                        )

                        if source not in displayed_sources:

                            displayed_sources.add(source)

                            st.write(
                                f"• {source}"
                            )

            except Exception as e:

                st.error(
                    "An error occurred while contacting the "
                    "Groq API or processing your question."
                )

                st.warning(
                    "Please check Streamlit → Manage app → Logs "
                    "for the detailed error."
                )

                # Show the error type without exposing secrets
                st.caption(
                    f"Error type: {type(e).__name__}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Zyro Dynamics HR Assistant • "
    "Powered by LangChain + FAISS + HuggingFace Embeddings + Groq"
)
```
