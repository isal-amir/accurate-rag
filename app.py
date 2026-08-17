import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import time
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.parser import parse_pdf
from src.vectorstore import ingest_parsed_pages
from src.graph import rag_app

st.set_page_config(page_title="Accurate Customer Support AI", page_icon="🧾", layout="wide")

st.title("Accurate RAG Chatbot (Support Assistant)")

# Sidebar for Knowledge Base
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if st.button("Ingest Document") and uploaded_file:
        with st.spinner("Parsing and Ingesting Document... (This may take a while)"):
            try:
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Parse PDF
                parsed_pages = parse_pdf(tmp_file_path)
                st.success(f"Parsed {len(parsed_pages)} pages.")
                
                # Ingest to Qdrant
                num_chunks = ingest_parsed_pages(parsed_pages)
                st.success(f"Ingested {num_chunks} chunks to vector store.")
                
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "summarized_count" not in st.session_state:
    st.session_state.summarized_count = 0

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Tanyakan sesuatu tentang Accurate..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            start_time = time.time()
            
            # --- MEMORY MANAGEMENT ---
            unsummarized_count = len(st.session_state.messages) - st.session_state.summarized_count
            
            # If > 10 unsummarized messages (5 turns), summarize the oldest 6 (3 turns)
            # This leaves 4 unsummarized messages (2 turns) prior to the new user prompt
            if unsummarized_count > 10:
                messages_to_summarize = st.session_state.messages[st.session_state.summarized_count : st.session_state.summarized_count + 6]
                
                summary_prompt = "Ringkas percakapan berikut ini menjadi satu paragraf padat. Fokus pada informasi penting dan konteks.\n\n"
                if st.session_state.summary:
                    summary_prompt += f"Ringkasan sebelumnya:\n{st.session_state.summary}\n\n"
                summary_prompt += "Percakapan baru:\n"
                for msg in messages_to_summarize:
                    role = "User" if msg["role"] == "user" else "AI"
                    summary_prompt += f"{role}: {msg['content']}\n"
                    
                summarizer_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
                summary_response = summarizer_llm.invoke([HumanMessage(content=summary_prompt)])
                st.session_state.summary = summary_response.content
                
                st.session_state.summarized_count += 6
            
            # Build history
            history = []
            if st.session_state.summary:
                history.append(SystemMessage(content=f"Ringkasan percakapan sebelumnya:\n{st.session_state.summary}"))
                
            # Exclude the current user prompt (which is the last message)
            unsummarized_msgs = st.session_state.messages[st.session_state.summarized_count : -1]
            for msg in unsummarized_msgs:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                else:
                    history.append(AIMessage(content=msg["content"]))

            inputs = {
                "question": prompt,
                "chat_history": history
            }
            
            try:
                # Run LangGraph
                output = rag_app.invoke(inputs)
                answer = output["generation"]
                
                end_time = time.time()
                latency = round(end_time - start_time, 2)
                
                # Append latency text
                st.markdown(answer)
                st.caption(f"⏱️ Waktu respons: {latency} detik")
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
