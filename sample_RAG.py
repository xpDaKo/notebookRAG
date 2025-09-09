from PyPDF2 import PdfReader # PDF文档读取、处理的依赖库
from langchain.text_splitter import RecursiveCharacterTextSplitter # LangChain封装的文档切分库
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder# MessagesPlaceholder占位符，
from langchain_community.vectorstores import FAISS # LangChain使用FAISS向量数据库保存切分后短文档的文本块向量
from langchain.tools.retriever import create_retriever_tool #RAG中的R，把RAG系统中的检索功能封装成工具，提供检索文本块向量功能
from langchain.agents import AgentExecutor, create_openai_functions_agent #LangChain中高层封装的Agent
from langchain_community.embeddings import DashScopeEmbeddings #调用阿里云百炼平台的Embedding模型
from langchain_community.chat_models import ChatTongyi as Tongyi

import os
from dotenv import load_dotenv
load_dotenv()

DASH_SCOPE_API_KEY = os.getenv("DASH_SCOPE_API_KEY")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASH_SCOPE_API_KEY
)
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASH_SCOPE_API_KEY)

def pdf_read(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks, save_dir="faiss_index"):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    os.makedirs(save_dir, exist_ok=True)   # 目录不存在就新建
    vectorstore.save_local(save_dir)
    print(f"✅ 向量库已保存到 ./{save_dir}")
    return vectorstore

def get_conversational_chain(tools, querys):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是AI助手，请根据检索结果回答问题，不要编造。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 必须
    ])
    tool = [tools]
    agent = create_openai_functions_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": querys})
    return response['output']

def check_database_exists():
    """检查FAISS数据库是否存在"""
    return os.path.exists(r"faiss_index") and os.path.exists(r"faiss_index\index.faiss")

def user_input(user_question):
    # 检查数据库是否存在
    if not check_database_exists():
        print('数据库不存在')
        return
    try:
        # 加载FAISS数据库
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever() 
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor","This tool is to give answer to queries from the pdf")
        return get_conversational_chain(retrieval_chain, user_question)
    except Exception as e:
        print(f'加载数据库时出错: {str(e)}')

def main():
    if not check_database_exists():
        print('数据库不存在, 开始创建数据库...')
        text_pdf = pdf_read(r'data\1-大模型（LLMs）基础面.pdf')
        text_chunks = get_chunks(text_pdf)
        vector_store(text_chunks)
        print('数据库创建成功!')
    
    return user_input('主流的开源模型体系分为哪几类？')
if __name__ == "__main__":
    main()
