import os
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.vectorstore import get_vectorstore
from dotenv import load_dotenv

load_dotenv()

# We use OpenRouter with LangChain's ChatOpenAI
def get_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="poolside/laguna-s-2.1:free",
        temperature=0
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    chat_history: List[Any]
    documents: List[Any]
    generation: str
    is_relevant: bool
    rewrite_count: int

def retrieve(state: GraphState):
    """
    Retrieve documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def evaluate(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---EVALUATE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"is_relevant": False}
        
    llm = get_llm()
    
    prompt = (
        "Anda adalah penilai untuk menentukan apakah dokumen yang diambil relevan dengan pertanyaan pengguna.\n"
        "Dokumen:\n"
        f"{[doc.page_content for doc in documents]}\n"
        f"Pertanyaan: {question}\n"
        "Jawab dengan 'ya' jika relevan dan mengandung informasi untuk menjawab pertanyaan, atau 'tidak' jika tidak."
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip().lower()
    
    if "ya" in answer:
        return {"is_relevant": True}
    else:
        return {"is_relevant": False}

def generate(state: GraphState):
    """
    Generate answer
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    
    context = ""
    for doc in documents:
        page = doc.metadata.get("page", "Unknown")
        context += f"[Halaman {page}]:\n{doc.page_content}\n\n"
        
    system_prompt = (
        "Anda adalah asisten AI untuk customer support software akuntansi online.\n"
        "Gunakan konteks dokumen dan ringkasan percakapan sebelumnya (jika ada) untuk menjawab pertanyaan pengguna.\n"
        "Jawaban harus dalam bahasa Indonesia.\n"
        "Sertakan sumber halaman (misal: 'Berdasarkan Halaman 2, ...') di dalam jawaban Anda.\n\n"
        f"Konteks Dokumen:\n{context}"
    )
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=question))
    
    llm = get_llm()
    response = llm.invoke(messages)
    
    return {"generation": response.content}

def rewrite(state: GraphState):
    """
    Rewrite the question to produce a better search query.
    """
    print("---REWRITE---")
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    
    llm = get_llm()
    
    msg = [
        SystemMessage(
            content="Anda adalah ahli pembuat query pencarian. Misi Anda adalah merumuskan ulang pertanyaan pengguna "
                    "agar lebih optimal untuk pencarian di database vektor (semantic search). "
                    "Fokus pada kata kunci utama dan hindari kata-kata yang tidak perlu. "
                    "Berikan hanya query hasil rumusan ulang Anda tanpa teks tambahan."
        ),
        HumanMessage(content=f"Pertanyaan awal: {question}")
    ]
    
    response = llm.invoke(msg)
    better_question = response.content.strip()
    print(f"---REWRITTEN QUERY: {better_question}---")
    
    return {"question": better_question, "rewrite_count": rewrite_count + 1}

def fallback(state: GraphState):
    """
    Fallback answer when documents are irrelevant.
    """
    print("---FALLBACK---")
    return {"generation": "Mohon maaf saya tidak tahu, informasi tidak ditemukan di basis pengetahuan."}

def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, rewrite, or use fallback.
    """
    is_relevant = state.get("is_relevant", False)
    rewrite_count = state.get("rewrite_count", 0)
    
    if is_relevant:
        return "generate"
    elif rewrite_count < 1:
        return "rewrite"
    else:
        return "fallback"

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("evaluate", evaluate)
workflow.add_node("generate", generate)
workflow.add_node("fallback", fallback)
workflow.add_node("rewrite", rewrite)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

rag_app = workflow.compile()
