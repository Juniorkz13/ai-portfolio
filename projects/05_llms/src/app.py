import streamlit as st
import requests

st.set_page_config(
    page_title="LLM RAG Assistente de Documentos",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LLM RAG Assistente de Documentos")
st.write("Faça perguntas sobre seus documentos usando um assistente de IA.")

API_URL = "http://localhost:8000"

st.subheader("📂 Ingestão de Documentos")

if st.button("Ingestionar Documentos"):
    with st.spinner("Indexando documentos..."):
        response = requests.post(f"{API_URL}/ingest")

        if response.status_code == 200:
            st.success("Documentos indexados com sucesso!")
        else:
            st.error("Erro ao indexar documentos.")

st.subheader("❓ Faça uma Pergunta")

question = st.text_input("Digite sua pergunta:")

if st.button("Perguntar"):
    if question.strip() == "":
        st.warning("Por favor, insira uma pergunta.")
    else:
        with st.spinner("Pensando..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question}
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
                st.markdown("### ✅ Resposta")
                st.write(answer)
            else:
                st.error("Erro ao obter resposta da API.")