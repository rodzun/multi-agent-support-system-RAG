from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
import os


langfuse = Langfuse()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL")
)


db_path = "vectorstores/hr_faiss"

if os.path.exists(db_path):
    span = langfuse.start_span(name="hr-faiss-load")
    vectorstore = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    span.end()

else:
    loader = DirectoryLoader(
        "data/hr_docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        recursive=True,
        silent_errors=True
    )

    span = langfuse.start_span(name="hr-load-docs")
    docs = loader.load()
    span.end()

    print("Docs loaded:", len(docs))
    if not docs:
        raise ValueError("No TXT files found inside data/hr_docs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    span = langfuse.start_span(name="hr-split-docs")
    chunks = splitter.split_documents(docs)
    span.end()

    print("Chunks:", len(chunks))
    if not chunks:
        raise ValueError("No chunks were created — your TXT file may be empty.")

    span = langfuse.start_span(name="hr-faiss-build")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path)
    span.end()

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL")
)

template = """You are an expert HR assistant. Answer based ONLY on the provided context.
If you don't know, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

hr_chain = (
    {
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
        "question": RunnableLambda(lambda x: x["question"])
    }
    | prompt
    | llm
    | StrOutputParser()
)
