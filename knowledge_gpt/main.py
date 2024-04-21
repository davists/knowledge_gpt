import os
import streamlit as st
from timeit import default_timer as timer
from datetime import timedelta
import pickle
import concurrent.futures
from itertools import zip_longest

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import open_local_file, read_file, docs_as_text
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm
from pathlib import Path
import glob

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-4-turbo"]
PERSPECTIVA_SOU_LOCADOR = 'Sou Locador'
PERSPECTIVA_SOU_LOCATARIO = 'Sou Locatário'
PROJECT_ROOT = Path(__file__).parent.resolve()

contract_analysis_steps = [
    "1. Resumo Geral do Contrato: Identifique o tipo de contrato e faça um resumo geral.",
    "2. Extração de Informações Chave: Extraia automaticamente dados essenciais como tipo de contrato, localização do imóvel, identificação das partes (locador e locatário), e intermediadores, se aplicável.",
    "3. Análise dos Termos Financeiros: Detalhe os termos financeiros, tais como valores de aluguel, impostos, seguros, e quaisquer outros encargos inclusos, com ênfase em datas de pagamento e condições do primeiro pagamento.",
    "4. Delineamento das Responsabilidades das Partes: Enumere e explique as responsabilidades de cada parte, incluindo mas não limitado a pagamento de aluguel, manutenção do imóvel e obrigações relacionadas a serviços públicos.",
    "5. Revisão de Multas e Penalidades: Revise as cláusulas de multas e penalidades, interpretando as condições de aplicação e as implicações legais dessas cláusulas.",
    "6. Integração da Legislação Aplicável e Doutrina: Cite e discuta a legislação relevante, como a Lei do Inquilinato, e integre conceitos doutrinários para explicar o impacto das normas legais nas obrigações contratuais.",
    "7. Inclusão de Jurisprudência Relevante: Apresente decisões judiciais que tratam de questões semelhantes às presentes no contrato, especialmente aquelas que influenciam a interpretação de cláusulas controversas.",
    "8. Análise Aprofundada de Cláusulas Específicas: Examine profundamente cláusulas críticas sobre benfeitorias, condições do imóvel, seguros e rescisão do contrato, discutindo as consequências legais do não cumprimento.",
    "9. Explanação sobre Arbitragem e Resolução de Disputas: Detalhe a cláusula de arbitragem e as opções para resolução de disputas estipuladas no contrato, explicando como os litígios devem ser resolvidos conforme o acordado.",
    "10. Estratégias de Prevenção de Riscos e Recomendações: Forneça recomendações para as partes gerenciarem suas responsabilidades e minimizarem riscos legais e financeiros, com sugestões para manter a conformidade com o contrato e a legislação vigente.",
    "11. Conclusão e Síntese: Apresente uma conclusão resumindo os pontos principais da análise, destacando os direitos e deveres críticos, identificando riscos e oferecendo uma avaliação final sobre a conformidade do contrato com as normas legais e as práticas de mercado.",
    #"12. Resultado da Análise em Tabela: Tabele sua análise separando em colunas por exemplo (cláusula transcrita, lei aplicável, parte afetada, risco)",
]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")
full_answers = ''
full_sources = ''


def get_prompt_template():
    return """
        Analise o _document_ conforme a _instrucao_ e responda em portugues.

        <_document_>
            {documento}
        </_document_>

        <_instrucao_>
        {instrucao}
        </_instrucao_>
    """


def concurrent_processing(_function, _list_arguments):
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(_function, argument) for argument in _list_arguments]

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Task generated an exception: {e}")        

    return results


def get_full_result(params):
    query = get_prompt_template()
    query = query.format(
            documento=params[1],
            instrucao=params[0].split(':')[1].strip()
        )

    llm = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )
   
    return [int(params[0].split(':')[0].strip().split('.')[0]), params[0].split(':')[0].strip(), result]


st.set_page_config(page_title="📖Contrato Revisado - Versão Alpha", page_icon="📖", layout="wide")
st.header("Contrato Revisado - Versão Alpha")

# Enable caching for expensive functions
bootstrap_caching()

openai_api_key = os.environ.get('OPENAI_API_KEY')

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt", "jpg", "png", "jpeg"],
    # help="Scanned documents are not supported yet!",
)

model: str = st.selectbox("Model", options=MODEL_LIST)  # type: ignore

return_all_chunks = True
show_full_doc = False

if not uploaded_file:
    st.stop()

try:
    upload_file_content = read_file(uploaded_file)
    chunked_files = []

    chunck_pkl = f"{PROJECT_ROOT}/chunked_files.pkl"
    if os.path.exists(chunck_pkl):
        # Open the file in binary read mode
        with open(chunck_pkl, 'rb') as file:
            # Unpickle the data
            chunked_files = pickle.load(file)
        # print(chunked_files)
    else:
        base_knowledge = glob.glob(f"{PROJECT_ROOT}/knowledge_base/*.txt")

        for _file in base_knowledge:
            file_base = open_local_file(_file)
            chunked_files.append(chunk_file(file_base, chunk_size=300, chunk_overlap=0))
            break

        # Open a file and use pickle.dump()
        with open(chunck_pkl, 'wb') as file:
            pickle.dump(chunked_files, file)
        
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name)

if not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

with st.spinner("Extraindo Texto do Documento ...⏳"):
    folder_index = embed_files(
        files=chunked_files,
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        openai_api_key=openai_api_key,
    )

if show_full_doc:
    with st.expander("Documento Enviado Para Analise"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(docs_as_text(upload_file_content.docs))}</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1,2])

answer_col, sources_col = st.columns(2)
start = timer()

with answer_col:
    st.markdown("#### Resposta (GPT4 + Fontes Internas)")
        
with sources_col:
    st.markdown("#### Fontes Internas Utilizadas")

with st.spinner("Gerando a Resposta ...⏳"):
    documento = docs_as_text(upload_file_content.docs)
    params = list(zip_longest(contract_analysis_steps, [documento], fillvalue=documento))
    results = concurrent_processing(get_full_result, params)
    results = sorted(results, key=lambda x: x[0])

    for result in results:
        with answer_col:
            st.markdown(f"**{result[1]}**")
            st.markdown(result[2].answer)

        with sources_col:
            for source in result[2].sources:
                st.markdown(source.page_content)
                st.markdown(source.metadata["source"])
                st.markdown("---")
        
end = timer()
time_elapsed = timedelta(seconds=end-start)
with answer_col:
    st.markdown(f"**Tempo de Processamento: {time_elapsed}**")  