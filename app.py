import streamlit as st
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --- 1. Ortam değişkenlerini yükle ---
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    st.error("❌ Lütfen `.env` dosyasına `OPENAI_API_KEY` anahtarını ekleyin.")
    st.stop()


# --- 2. RAG pipeline (önbelleğe alınmış) ---
@st.cache_resource
def build_rag_system():
    pdf_path = "data/analog-elektronik.pdf"

    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' bulunamadı. Lütfen PDF dosyasını 'data' klasörüne ekleyin.")
        st.stop()

    # PDF yükle
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Metni parçalara ayır
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Vektör veritabanı oluştur
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Model ve prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
Aşağıdaki bağlamdan yararlanarak kullanıcı sorusunu yanıtla.
Bağlam yetersizse "Bu bilgi belgeden çıkarılamıyor." de.

Kullanıcı ile önceki konuşma geçmişi:
{chat_history}

Bağlam:
{context}

Soru:
{question}
""")

    return retriever, llm, prompt


# --- 3. Cevap oluşturma fonksiyonu ---
def create_answer(question, chat_history, retriever, llm, prompt):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "question": question,
        "context": context,
        "chat_history": "\n".join([f"Kullanıcı: {q}\nBot: {a}" for q, a in chat_history])
    })


# --- 4. Streamlit arayüzü ---
def main():
    st.set_page_config(page_title="Elektrik ve Elektronik RAG Asistanı", page_icon="⚡")
    st.title("⚡ Elektrik ve Elektronik Ders Notları Asistanı")
    #st.caption("Veri kaynağı: `analog-elektronik.pdf`")
    st.markdown("Yüklenen tüm Elektrik ve Elektronik ders notlarını kullanarak bilgiye dayalı (Grounded RAG) yapay zeka sistemi ile etkileşim kurun. Bu asistan, yalnızca sağlanan ders materyallerine göre cevap üretir.")
    retriever, llm, prompt = build_rag_system()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Sorunuzu yazın:", placeholder="Örneğin: Diyot nasıl çalışır?")

    if user_input:
        with st.spinner("Yanıt hazırlanıyor..."):
            answer = create_answer(user_input, st.session_state.chat_history, retriever, llm, prompt)
            st.session_state.chat_history.append((user_input, answer))
            st.markdown(f"**🤖 Asistan:** {answer}")


if __name__ == "__main__":
    main()
