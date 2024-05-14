# flake8: noqa
from langchain.prompts import PromptTemplate

# Use a shorter template to reduce the number of tokens in the prompt
template = """Gere uma minuta de escritura pública de compra e venda use como referencia o _document_ preencha os dados pessoais com as informações em _dados_basicos_.
Crie uma resposta final em português retornando apenas o documento gerado com texto formatado destacando titulos e subtitulos. Cada linha de cada paragrafo deve contem no maximo 120 caracteres.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:
"""


STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)
