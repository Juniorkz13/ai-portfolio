from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT_SECONDS = 60
FIXED_TOP_K = 5


def apply_theme() -> None:
    """Apply a polished single light theme for a professional SaaS-like UI."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #f6f8fb;
                --bg-soft: #f9fbfe;
                --surface: #ffffff;
                --surface-alt: #f8fafc;
                --border: #e5e7eb;
                --text: #1f2933;
                --text-muted: #6b7280;
                --primary: #2563eb;
                --primary-hover: #1d4ed8;
                --primary-soft: #eff6ff;
                --success: #16a34a;
                --success-soft: #f0fdf4;
                --error: #dc2626;
                --error-soft: #fef2f2;
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, #ffffff 0, #f6f8fb 45%),
                    var(--bg);
                color: var(--text);
            }

            .block-container {
                max-width: 1140px;
                padding-top: 1.4rem;
                padding-bottom: 2.3rem;
            }

            h1, h2, h3, h4, h5, h6, p, label, span, div {
                color: var(--text);
            }

            .hero {
                background: linear-gradient(145deg, #ffffff 0%, #f8fbff 100%);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.15rem 1.2rem 1.1rem 1.2rem;
                margin-bottom: 1rem;
            }

            .hero-title {
                margin: 0;
                font-size: clamp(1.65rem, 2.1vw, 2.2rem);
                line-height: 1.23;
                font-weight: 680;
                letter-spacing: -0.01em;
                white-space: normal;
                overflow: visible;
                text-wrap: balance;
                word-break: keep-all;
            }

            .hero-subtitle {
                margin-top: 0.42rem;
                margin-bottom: 0;
                color: var(--text-muted);
                font-size: 0.96rem;
                line-height: 1.45;
                max-width: 78ch;
            }

            .section-card {
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 1rem 1.05rem;
                margin-bottom: 1rem;
            }

            .section-helper {
                color: var(--text-muted);
                font-size: 0.91rem;
                margin-bottom: 0.55rem;
            }

            .doc-card,
            .source-card {
                background: var(--surface-alt);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.9rem 0.95rem;
                margin-bottom: 0.65rem;
            }

            .doc-title,
            .source-title {
                margin: 0 0 0.28rem 0;
                font-size: 0.98rem;
                font-weight: 600;
                color: var(--text);
            }

            .chip {
                display: inline-block;
                padding: 0.18rem 0.5rem;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: var(--primary-soft);
                color: #3b4a5a;
                font-size: 0.75rem;
                margin-right: 0.4rem;
                margin-bottom: 0.4rem;
            }

            .response-primary {
                background: #ffffff;
                border: 1px solid #dbe6fb;
                border-radius: 12px;
                padding: 0.95rem;
                margin-bottom: 0.75rem;
            }

            .response-secondary {
                background: var(--surface-alt);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.95rem;
                margin-bottom: 0.8rem;
            }

            .empty-state {
                background: var(--bg-soft);
                border: 1px dashed var(--border);
                border-radius: 12px;
                padding: 0.85rem 0.9rem;
                color: var(--text-muted);
                font-size: 0.93rem;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                border-radius: 10px;
                border: 1px solid var(--primary);
                background: var(--primary);
                color: #ffffff;
                font-weight: 520;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                background: var(--primary-hover);
                border-color: var(--primary-hover);
                color: #ffffff;
            }

            .stTextInput input,
            .stTextArea textarea,
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                background: var(--surface-alt) !important;
                color: var(--text) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
            }

            .stTextInput input:focus,
            .stTextArea textarea:focus {
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 1px var(--primary) !important;
            }

            .stFileUploader > div {
                background: var(--surface-alt);
                border: 1px dashed #d7deea;
                border-radius: 10px;
            }

            [data-testid="stSidebar"] {
                background: var(--surface);
                border-right: 1px solid var(--border);
            }

            [data-testid="stAlert"] {
                border-radius: 10px;
                border: 1px solid var(--border);
            }

            [data-testid="stAlert"][kind="success"] {
                background: var(--success-soft);
                color: var(--success);
            }

            [data-testid="stAlert"][kind="error"] {
                background: var(--error-soft);
                color: var(--error);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def request_api(
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[bool, Any, str | None]:
    """Execute an HTTP request and return `(ok, payload, error_message)`."""
    try:
        response = requests.request(
            method=method,
            url=url,
            json=json,
            files=files,
            data=data,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return False, None, f"Não foi possível conectar à API: {exc}"

    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    if response.status_code >= 400:
        if isinstance(payload, dict) and payload.get("detail"):
            return False, payload, str(payload["detail"])
        return False, payload, f"Erro HTTP {response.status_code}"

    return True, payload, None


def fetch_documents(api_base_url: str, show_errors: bool = True) -> list[dict[str, Any]]:
    """Fetch catalog documents from backend API."""
    ok, payload, error = request_api("GET", f"{api_base_url}/documents")
    if not ok:
        if show_errors:
            st.error(error or "Não foi possível carregar os documentos.")
        return []
    if not isinstance(payload, list):
        if show_errors:
            st.error("A API retornou um formato inesperado para a listagem de documentos.")
        return []
    return payload


def render_header() -> None:
    """Render application header."""
    st.markdown(
        (
            '<div class="hero">'
            '<h1 class="hero-title">Assistente de Normas Técnicas</h1>'
            '<p class="hero-subtitle">Consulte normas técnicas com respostas fundamentadas em documentos, fontes rastreáveis e gestão simples do acervo.</p>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_upload(api_base_url: str) -> None:
    """Render PDF upload workflow."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload de documento")
    st.markdown('<div class="section-helper">Envie um PDF para indexação no acervo técnico.</div>', unsafe_allow_html=True)

    with st.form("upload_form", clear_on_submit=True):
        pdf_file = st.file_uploader("Arquivo PDF", type=["pdf"])
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input("Título (opcional)", placeholder="Ex.: IT-01 Segurança contra Incêndio")
        with col2:
            document_type = st.text_input("Tipo de documento", value="unknown", placeholder="Ex.: fire_regulation")
        with col3:
            version = st.text_input("Versão", value="1.0", placeholder="Ex.: 2026")
        submitted = st.form_submit_button("Enviar documento")

    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        return
    if pdf_file is None:
        st.warning("Selecione um arquivo PDF antes de enviar.")
        return

    files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
    data = {
        "title": title or "",
        "document_type": document_type,
        "version": version,
    }

    with st.spinner("Processando upload e indexação do documento..."):
        ok, payload, error = request_api("POST", f"{api_base_url}/upload", files=files, data=data)
    if not ok:
        st.error(error or "Não foi possível concluir o upload e processamento do documento.")
        return

    st.success("Documento enviado e processado com sucesso.")
    if isinstance(payload, dict):
        st.caption(
            f"ID: {payload.get('document_id')} | "
            f"Páginas: {payload.get('total_pages')} | "
            f"Chunks: {payload.get('total_chunks')}"
        )


def render_documents(api_base_url: str) -> list[dict[str, Any]]:
    """Render document catalog and deletion actions."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_title, col_action = st.columns([4, 1])
    with col_title:
        st.subheader("Acervo carregado")
        st.markdown('<div class="section-helper">Visualize e remova documentos indexados.</div>', unsafe_allow_html=True)
    with col_action:
        if st.button("Atualizar", use_container_width=True):
            st.rerun()

    documents = fetch_documents(api_base_url)
    if not documents:
        st.markdown('<div class="empty-state">Nenhum documento disponível no momento.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return []

    for document in documents:
        title = document.get("title", "Sem título")
        st.markdown(
            f'<div class="doc-card">'
            f'<p class="doc-title">#{document["id"]} · {title}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="chip">tipo: {document.get("document_type", "-")}</span>'
            f'<span class="chip">versão: {document.get("version", "-")}</span>'
            f'<span class="chip">chunks: {document.get("total_chunks", "-")}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Arquivo: {document.get('filename', '-')}  |  Upload: {document.get('uploaded_at', '-')}")

        col1, col2 = st.columns([2, 1])
        with col1:
            remove_file = st.checkbox(
                "Remover arquivo físico junto",
                value=True,
                key=f"remove_file_{document['id']}",
            )
        with col2:
            if st.button("Remover", key=f"delete_doc_{document['id']}", use_container_width=True):
                with st.spinner("Removendo documento do acervo..."):
                    ok, _, error = request_api(
                        "DELETE",
                        f"{api_base_url}/documents/{document['id']}",
                        params={"remove_file": str(remove_file).lower()},
                    )
                if not ok:
                    st.error(error or "Não foi possível remover o documento.")
                else:
                    st.success("Documento removido com sucesso do acervo.")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    return documents


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render enriched source blocks in a concise professional layout."""
    st.markdown("#### Fontes utilizadas")
    if not sources:
        st.markdown('<div class="empty-state">Nenhuma fonte retornada para esta consulta.</div>', unsafe_allow_html=True)
        return

    for source in sources:
        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="source-title">{source.get("document_title", "Documento")}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="chip">tipo: {source.get("document_type", "-")}</span>'
            f'<span class="chip">versão: {source.get("version", "-")}</span>'
            f'<span class="chip">página: {source.get("page_number", "-")}</span>',
            unsafe_allow_html=True,
        )
        st.write(source.get("excerpt", "-"))
        st.markdown('</div>', unsafe_allow_html=True)


def render_chat(api_base_url: str, documents: list[dict[str, Any]]) -> None:
    """Render chat form and response panel."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Consulta técnica")
    st.markdown(
        '<div class="section-helper">Faça perguntas sobre normas técnicas e refine com filtros opcionais.</div>',
        unsafe_allow_html=True,
    )

    document_ids = [str(doc["id"]) for doc in documents]
    document_types = sorted({doc.get("document_type", "") for doc in documents if doc.get("document_type")})
    versions = sorted({doc.get("version", "") for doc in documents if doc.get("version")})

    with st.form("chat_form"):
        question = st.text_area(
            "Pergunta",
            height=130,
            placeholder="Ex.: Qual a largura mínima exigida para escadas de emergência em edifícios de uso coletivo?",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_document_id = st.selectbox(
                "Documento (ID) - opcional",
                options=[""] + document_ids,
                index=0,
            )
        with col2:
            selected_document_type = st.selectbox(
                "Tipo de documento - opcional",
                options=[""] + document_types,
                index=0,
            )
        with col3:
            selected_version = st.selectbox(
                "Versão - opcional",
                options=[""] + versions,
                index=0,
            )

        submitted = st.form_submit_button("Gerar resposta")

    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        return
    if not question.strip():
        st.warning("Digite uma pergunta para iniciar a consulta.")
        return

    payload: dict[str, Any] = {
        "question": question,
        "top_k": FIXED_TOP_K,
    }
    if selected_document_id:
        payload["document_id"] = int(selected_document_id)
    if selected_document_type:
        payload["document_type"] = selected_document_type
    if selected_version:
        payload["version"] = selected_version

    with st.spinner("Consultando o acervo e gerando resposta técnica..."):
        ok, response_payload, error = request_api("POST", f"{api_base_url}/chat", json=payload)
    if not ok:
        st.error(error or "Não foi possível processar a consulta no chat.")
        return
    if not isinstance(response_payload, dict):
        st.error("Resposta inesperada da API no endpoint /chat.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Resultado da consulta")
    st.markdown('<div class="response-primary">', unsafe_allow_html=True)
    st.markdown("**Resposta objetiva**")
    st.write(response_payload.get("answer", "-"))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="response-secondary">', unsafe_allow_html=True)
    st.markdown("**Explicação técnica**")
    st.write(response_payload.get("explanation", "-"))
    st.markdown("</div>", unsafe_allow_html=True)

    render_sources(response_payload.get("sources", []))
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    """Run Streamlit MVP UI."""
    st.set_page_config(page_title="Assistente de Normas Técnicas", layout="wide")
    apply_theme()

    with st.sidebar:
        st.markdown("### Configuração")
        api_base_url = st.text_input("URL da API", value=DEFAULT_API_BASE_URL).rstrip("/")
        st.caption("Aponte para a API FastAPI em execução local.")
        st.caption("A busca usa profundidade fixa interna para manter a interface simples.")

    render_header()

    tab_chat, tab_docs, tab_upload = st.tabs(["Consulta", "Acervo", "Upload"])

    with tab_chat:
        docs_for_filters = fetch_documents(api_base_url, show_errors=False)
        render_chat(api_base_url, docs_for_filters)

    with tab_docs:
        render_documents(api_base_url)

    with tab_upload:
        render_upload(api_base_url)


if __name__ == "__main__":
    main()
