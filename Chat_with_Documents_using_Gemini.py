import os
import tempfile
import streamlit as st
import requests  # API 호출을 위한 라이브러리
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit 페이지 설정
st.set_page_config(
    page_title="LangChain: Chat with Documents",
    page_icon="🦜"  # 이모지를 직접 사용
)
st.title("🦜 LangChain: Chat with Documents using Gemini")

# 문서 검색기 설정 함수
@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

# 스트리밍 출력을 Streamlit으로 보내는 커스텀 콜백 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, run_id=None, **kwargs):
        if run_id is None:
            run_id = "default_run_id"
        
    # 기존 로직 유지
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Gemini API 키를 Streamlit 사이드바에서 읽기
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.")
    st.stop()

# PDF 파일 업로드
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Google Gemini 모델을 사용하는 LLM 클래스
class GeminiLLM:
    def __init__(self, api_key, endpoint="https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.0-flash:generateText", temperature=0.7):
        self.api_key = api_key
        self.endpoint = endpoint
        self.temperature = temperature

    def generate_response(self, query):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": query,
            "temperature": self.temperature,
            "maxOutputTokens": 256  # 필요에 따라 조정
            # 추가 설정 예시:
            # "top_p": 0.95,
            # "top_k": 50
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("candidates", [{"output": "No response"}])[0]["output"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Gemini API: {e}"

# 대화형 검색 체인 정의
class GeminiConversationalRetrievalChain:
    def __init__(self, retriever, llm, memory):
        self.retriever = retriever
        self.llm = llm
        self.memory = memory

    def run(self, query, callbacks=None):
        # 수정된 메서드 호출
        retrieved_docs = self.retriever.get_relevant_documents(query)
        for callback in callbacks or []:
            callback.on_retriever_end(retrieved_docs, run_id="custom_run_id")

            response = self.llm.generate_response(query)
        return response


gemini_llm = GeminiLLM(api_key=gemini_api_key)
qa_chain = GeminiConversationalRetrievalChain(retriever=retriever, llm=gemini_llm, memory=memory)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
