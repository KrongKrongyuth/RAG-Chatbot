from langchain_ollama.llms import OllamaLLM # LLMs
from langchain_huggingface import HuggingFaceEmbeddings # Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain import hub
import gradio as gr

llm_model: str = "llama3.1:8b"
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

llm = OllamaLLM(model=llm_model)
embedding = HuggingFaceEmbeddings(model_name=embedding_model)
rag_prompt = hub.pull("rlm/rag-prompt")
        
def get_response(query: str, history: str = None) -> str:
    """`
    Get a response from the Ollama LLM for a given query.

    Args:
        query (str): The input query to send to the LLM.

    Returns:
        str: The response from the LLM.
    """
    retriever = FAISS.load_local('vectorDB', embedding, allow_dangerous_deserialization=True).as_retriever()
    
    retrieved_doc = retriever.invoke(query, search_kwargs={"k": 3})
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_doc)
    
    message = rag_prompt.invoke({'question':query, 'context':docs_content})
    response = llm.invoke(message)
    
    return response

def pdf_vecDB(file_path: str) -> None:
    """
    Transfroms pdf file into FAISS vectorDB

    Args:
        file_path (str): PDF file path
        save_loc (str): Save location
    """
    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()
    print(pages)
    
    faiss_db = FAISS.from_documents(documents=pages,
                                    embedding=embedding)
    faiss_db.save_local('vectorDB')

if __name__ == '__main__':
    gr.ChatInterface(
        fn=get_response, 
        type="messages"
    ).launch()