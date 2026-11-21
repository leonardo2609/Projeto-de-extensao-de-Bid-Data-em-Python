<div align="center">

# 🛣️ Análise de Big Data: Acidentes nas Rodovias Federais (PRF 2007–2026)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Big Data](https://img.shields.io/badge/Big%20Data-ETL%20%26%20Analytics-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen?style=for-the-badge)

**Projeto de Extensão Universitária focado em Engenharia, Tratamento e Análise de Dados em Larga Escala utilizando Python.**

[Sobre o Projeto](#-sobre-o-projeto) • [Arquitetura e Dados](#-arquitetura-e-dados) • [Estrutura do Repositório](#-estrutura-do-repositório) • [Pipeline ETL](#-pipeline-etl-e-tratamento) • [Como Executar](#-como-executar)

</div>

---

## 📖 Sobre o Projeto

Este repositório contém o desenvolvimento completo de um **Projeto de Extensão em Big Data**, cujo objetivo principal é coletar, higienizar, consolidar e analisar uma série histórica de **20 anos (2007 a 2026)** dos registros de acidentes de trânsito nas rodovias federais brasileiras, disponibilizados pela **Polícia Rodoviária Federal (PRF)**.

Lidando com um volume superior a **500 MB de dados brutos** distribuídos em dezenas de arquivos anuais, o projeto demonstra na prática como aplicar técnicas de processamento de dados em Python para transformar registros heterogêneos e massivos em uma base unificada e otimizada para geração de *insights* e apoio à tomada de decisão.

### 🎯 Objetivos Principais
* **Ingestão em Larga Escala:** Manipulação eficiente de 20 bases anuais distintas (`datatran2007` a `datatran2026`).
* **Padronização e Limpeza (Data Wrangling):** Tratamento de inconsistências de esquema (mudanças de colunas ao longo das décadas), valores nulos, tipagem e codificação de caracteres.
* **Redução e Otimização:** Filtragem estratégica de variáveis relevantes, reduzindo o volume bruto (~498 MB) para uma base consolidada altamente performática (~3.1 MB).
* **Análise Exploratória e Visualização:** Extração de métricas sobre sazonalidade, causas de acidentes, gravidade e distribuição geográfica.

---

## 📊 Arquitetura e Dados

Os dados utilizados são provenientes do portal de Dados Abertos da **Polícia Rodoviária Federal (Datatran)**. Devido à longa janela temporal (2007–2026), os arquivos originais apresentam variações estruturais significativas, exigindo um pipeline robusto de normalização.

| Camada de Dados | Diretório / Arquivo | Volume Aproximado | Descrição |
| :--- | :--- | :---: | :--- |
| **Bronze (Raw)** | `PRF - Dados Brutos/` | ~498.6 MB | 20 arquivos CSV originais anuais sem alterações. |
| **Silver (Cleaned)** | `PRF - Dados Tratados/` | ~3.2 MB | 20 arquivos CSV higienizados, filtrados e padronizados ano a ano. |
| **Gold (Consolidated)** | `datatran_consolidado.csv` | ~3.1 MB | Base única consolidada pronta para análises e dashboards. |

---

## 📂 Estrutura do Repositório

```text
Projeto-de-extensao-de-Bid-Data-em-Python/
│
├── 📁 PRF - Dados Brutos/              # Bases originais (datatran2007.csv ... datatran2026.csv)
├── 📁 PRF - Dados Tratados/            # Bases anuais após limpeza (datatran2007_tratado.csv ...)
├── 📁 .vscode/                         # Configurações do ambiente de desenvolvimento
│
├── 📄 datatran_consolidado.csv         # Dataset final unificado (2007 a 2026)
├── 📓 script.ipynb                     # Jupyter Notebook com o passo a passo e análises visuais
├── 🐍 script.py                        # Script Python automatizado para execução do pipeline ETL
├── 📊 Slide_Big_Data.pdf               # Apresentação executiva dos resultados e metodologia
└── 📝 README.md                        # Documentação do projeto
```

---

## ⚙️ Pipeline ETL e Tratamento

O fluxo de processamento implementado nos arquivos `script.py` e `script.ipynb` segue as etapas clássicas de Engenharia de Dados:

1. **Extração (Extract):** Leitura iterativa dos arquivos CSV anuais na pasta `PRF - Dados Brutos/`, lidando com diferentes delimitadores (`;`) e encodings (`latin-1` / `utf-8`).
2. **Transformação (Transform):**
   * Normalização dos nomes das colunas entre os anos mais antigos (2007–2016) e mais recentes (2017–2026).
   * Tratamento de datas, horários e padronização de categorias (ex: causas de acidentes, tipos de pista, condições meteorológicas).
   * Remoção de registros corrompidos ou duplicados e seleção das colunas essenciais para o escopo analítico.
3. **Carga (Load):**
   * Exportação individual dos arquivos tratados para `PRF - Dados Tratados/`.
   * Concatenação de toda a série histórica no arquivo mestre `datatran_consolidado.csv`.

---

## 🚀 Como Executar

### Pré-requisitos
Certifique-se de ter o **Python 3.8+** instalado em sua máquina. Recomenda-se o uso de um ambiente virtual (`venv`).

```bash
# Clone este repositório
git clone https://github.com/seu-usuario/Projeto-de-extensao-de-Bid-Data-em-Python.git

# Acesse a pasta do projeto
cd Projeto-de-extensao-de-Bid-Data-em-Python

# Crie e ative um ambiente virtual (Opcional, mas recomendado)
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

### Instalação das Dependências
Instale as principais bibliotecas utilizadas na manipulação e análise dos dados:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Executando o Pipeline
Você pode rodar o processamento de duas formas:

* **Via Script Python (Execução direta):**
  ```bash
  python script.py
  ```
* **Via Jupyter Notebook (Análise interativa):**
  ```bash
  jupyter notebook script.ipynb
  ```

---

## 📑 Material de Apoio

Para uma visão executiva da metodologia aplicada, arquitetura da solução e os principais indicadores descobertos na análise dos 20 anos de dados, consulte a apresentação disponível diretamente no repositório:

👉 **[Visualizar Apresentação (Slide_Big_Data.pdf)](./Slide_Big_Data.pdf)**

---

<div align="center">
  <sub>Desenvolvido como parte do Projeto de Extensão em Big Data com Python.</sub>
</div>
