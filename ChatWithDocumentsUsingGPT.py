# 필요한 라이브러리 임포트
import os  # 파일 작업을 위한 라이브러리
import tempfile  # 임시 파일 저장을 위한 라이브러리
import streamlit as st  # Streamlit을 사용해 웹 앱 구축
from langchain.chat_models import ChatOpenAI  # OpenAI 채팅 모델 사용, + pip install --upgrade langchain openai
from langchain.document_loaders import PyPDFLoader  # PDF 문서 로드
from langchain.memory import ConversationBufferMemory  # 대화 기록 관리를 위한 메모리
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory  # Streamlit에서 메시지 저장
from langchain.embeddings import HuggingFaceEmbeddings  # 텍스트 임베딩 (Hugging Face 모델 사용)
from langchain.callbacks.base import BaseCallbackHandler  # 체인의 작업 중 콜백 핸들링
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.vectorstores import DocArrayInMemorySearch  # 메모리 내 문서 검색
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 작은 조각으로 나누기

# Streamlit 페이지 설정
st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")  # 페이지 제목과 아이콘 설정
st.title("🦜 LangChain: Chat with Documents")  # 페이지 제목 표시

# 문서 검색기 설정 함수
@st.cache_resource(ttl="1h")  # 1시간 동안 검색기를 캐시하여 재계산 방지
def configure_retriever(uploaded_files):
    # 업로드된 문서를 읽고 처리하기
    docs = []  # 로드된 문서를 저장할 리스트
    temp_dir = tempfile.TemporaryDirectory()  # 임시 디렉터리 생성
    for file in uploaded_files:  # 업로드된 파일마다 반복
        temp_filepath = os.path.join(temp_dir.name, file.name)  # 임시 파일 경로 생성
        with open(temp_filepath, "wb") as f:  # 파일을 임시로 저장
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)  # PDF 파일을 로드하는 로더 생성
        docs.extend(loader.load())  # 로드된 내용을 docs 리스트에 추가

    # 문서를 작은 조각으로 나누기
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  # 크기와 중복 설정
    splits = text_splitter.split_documents(docs)  # 문서들을 작은 조각으로 분할

    # HuggingFace 모델을 사용해 임베딩 생성
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 특정 HuggingFace 임베딩 모델 사용
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)  # 임베딩을 사용하여 벡터 DB 생성
       # 관련 문서를 검색하는 리트리버 정의
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})  # 검색 전략 설정

    return retriever  # 설정된 리트리버 반환


# 스트리밍 출력을 Streamlit으로 보내는 커스텀 콜백 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container  # 결과를 표시할 Streamlit 컨테이너
        self.text = initial_text  # 컨테이너에 표시할 초기 텍스트
        self.run_id_ignore_token = None  # 특정 실행을 무시할 토큰

    # LLM 시작 시 호출되는 메서드
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"):  # 사람이 입력한 질문을 출력하지 않기 위해
            self.run_id_ignore_token = kwargs.get("run_id")  # 실행 ID를 저장하여 특정 실행을 무시

    # LLM이 새로운 토큰을 생성할 때마다 호출되는 메서드
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):  # 특정 실행을 무시
            return
        self.text += token  # 토큰을 텍스트에 추가
        self.container.markdown(self.text)  # Streamlit 컨테이너에 새로운 텍스트 업데이트


# 컨텍스트 검색 상태를 출력하는 커스텀 콜백 핸들러
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")  # Streamlit에 상태 표시 설정

    # 리트리버가 작동을 시작할 때 호출되는 메서드
    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")  # 처리 중인 질문 표시
        self.status.update(label=f"**Context Retrieval:** {query}")  # 상태 레이블 업데이트

    # 리트리버가 작업을 완료할 때 호출되는 메서드
    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):  # 검색된 각 문서에 대해 반복
            source = os.path.basename(doc.metadata["source"])  # 문서의 출처(파일 이름) 가져오기
            self.status.write(f"**Document {idx} from {source}**")  # 표시되는 문서에 대해 알림
            self.status.markdown(doc.page_content)  # 문서 내용 표시
        self.status.update(state="complete")  # 검색 작업이 완료되었음을 표시


# OpenAI API 키를 Streamlit 사이드바에서 읽기
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")  # API 키를 안전하게 입력
if not openai_api_key:  # API 키가 제공되지 않으면
    st.info("Please add your OpenAI API key to continue.")  # API 키 입력을 요청하는 메시지 표시
    st.stop()  # API 키가 없으면 실행 중단

# PDF 파일 업로드
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)  # 여러 개의 PDF 파일 업로드 허용
if not uploaded_files:  # 파일이 업로드되지 않으면
    st.info("Please upload PDF documents to continue.")  # PDF 파일을 업로드할 것을 요청하는 메시지 표시
    st.stop()  # 파일이 없으면 실행 중단

# 업로드된 파일로 문서 검색기 설정
retriever = configure_retriever(uploaded_files)

# 대화 기록을 저장할 메모리 설정
msgs = StreamlitChatMessageHistory()  # 메시지 기록을 저장할 객체 생성
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)  # 대화 기록용 메모리 설정

# 채팅을 위한 언어 모델(LLM) 설정
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)  # 채팅을 위한 GPT 모델 인스턴스 생성
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)  # 대화형 검색 체인 설정

# 메시지 기록을 초기화하려면
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):  # 메시지가 없거나 버튼 클릭 시
    msgs.clear()  # 메시지 기록 초기화
    msgs.add_ai_message("How can I help you?")  # 기본 메시지 추가

# 채팅 인터페이스에 이전 메시지 표시
avatars = {"human": "user", "ai": "assistant"}  # 사용자와 AI의 아바타 설정
for msg in msgs.messages:  # 메시지 기록을 반복하면서
    st.chat_message(avatars[msg.type]).write(msg.content)  # 해당하는 아바타로 메시지 표시

# 사용자 입력을 처리하고 응답 생성
if user_query := st.chat_input(placeholder="Ask me anything!"):  # 사용자 입력 대기
    st.chat_message("user").write(user_query)  # 사용자 질문 표시

    with st.chat_message("assistant"):  # AI 응답 표시
        retrieval_handler = PrintRetrievalHandler(st.container())  # 컨텍스트 검색 핸들러 설정
        stream_handler = StreamHandler(st.empty())  # 스트림 핸들러 설정
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])  # QA 체인 실행