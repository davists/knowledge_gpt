import os
import streamlit as st
import pickle

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file, docs_as_text
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from pathlib import Path

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = [
    'ESCRITURA', 
    #'CONTRATO DE LOCAÇÃO DE IMÓVEL'
]
PROJECT_ROOT = Path(__file__).parent.resolve()
GPT_MODEL = 'gpt-4-turbo'

html_margin = """
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        margin-top: 10px;
    }
    .Widget>label {
        font-family: monospace;
    }
    [class^="st-b"]  {
        font-family: monospace;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-eb {
        margin: 10px 0px;
    }
    </style>
    """

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

def get_prompt_template():
    return """

        <_document_>
            {documento}
        </_document_>

        <_dados_basicos_>
            {dados_basicos}
        </_dados_basicos_>
    """

app_title = 'Elaboração de Contrato - Versão Alpha'
st.set_page_config(page_title=app_title, page_icon="📖", layout="wide")
st.header(app_title)

# Enable caching for expensive functions
bootstrap_caching()

model: str = st.selectbox("ELABORAR DOCUMENTO", options=MODEL_LIST)  # type: ignore
st.markdown(html_margin, unsafe_allow_html=True)
MODEL_LIST = [
    'ESCRITURA',        
    # 'CONTRATO DE LOCAÇÃO DE IMÓVEL'
]

if model == MODEL_LIST[0]:
    template = """
        COMPRADOR: JOAO SILVA
        CPF: 123.456.789-00

        VENDEDOR: MARIA SOUZA
        CPF: 987.654.321-00

        VALOR DO IMOVEL: R$ 100.000,00

        ...
    """
   
# if model == MODEL_LIST[1]:
#     template = """
#         LOCADOR: JOAO SILVA
#         CPF: 123.456.789-00

#         LOCATÁRIO: MARIA SOUZA
#         CPF: 987.654.321-00

#         VALOR DO ALUGUEL: R$ 100.000,00

#         ...
#     """
   
if model:
    # st.markdown('RESUMO DAS INFORMAÇÕES RELEVANTES SOBRE O DOCUMENTO')
    st.markdown(html_margin, unsafe_allow_html=True)
    dados_basicos = st.text_area('RESUMO DAS INFORMAÇÕES RELEVANTES SOBRE O DOCUMENTO', value=template, height=250)   
    criar_documento = st.button('GERAR DOCUMENTO')

    if criar_documento:
        with st.spinner("Gerando o Documento ...⏳"):
            if model == MODEL_LIST[0]:
                template_file = open(f"{PROJECT_ROOT}/elaboracao/minuta_pagamento_a_vista.pdf", 'rb')
                file_name = 'escritura.pdf'

            # if model == MODEL_LIST[1]:
            #     template_file = open(f"{PROJECT_ROOT}/elaboracao/contrato_aluguel_modelo.pdf", 'rb')
            #     file_name = 'contrato_aluguel.pdf'                

            upload_file_content = read_file(template_file)
            documento = docs_as_text(upload_file_content.docs)

            query = get_prompt_template()
            query = query.format(
                documento=documento,
                dados_basicos=dados_basicos
            )
      
            llm = get_llm(model=GPT_MODEL, temperature=0)
            chunck_pkl = f"{PROJECT_ROOT}/chunked_files.pkl"
            if os.path.exists(chunck_pkl):
                with open(chunck_pkl, 'rb') as file:
                    chunked_files = pickle.load(file)

            folder_index = embed_files(
                files=chunked_files,
                embedding=EMBEDDING if model != "debug" else "debug",
                vector_store=VECTOR_STORE if model != "debug" else "debug",
            )
            result = query_folder(
                folder_index=folder_index,
                query=query,
                return_all=False,
                llm=llm,
            )

            # Print HTML
            st.markdown("ESCRITURA (PAGAMENTO À VISTA)")
            st.markdown(result.answer)
