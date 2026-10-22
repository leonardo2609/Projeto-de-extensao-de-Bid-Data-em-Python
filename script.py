import pandas as pd
import unicodedata as uni
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def remove_acents(texto):
    if isinstance(texto, str):
        nfkd = uni.normalize('NFKD', texto)
        return u"".join([c for c in nfkd if not uni.combining(c)])
    return texto

def limpeza_e_processamento(arquivo):
    try:
        # Limpeza dos dados para padronização do dataframe
        if arquivo.shape[1] <= 26:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'ano', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'id'
            ], errors='ignore')
        else:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'ano', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'latitude', 'longitude', 'regional', 'delegacia', 'uop', 'id'
            ], errors='ignore')


        # Aplicação das funções de texto
        arquivo['causa_acidente'] = arquivo['causa_acidente'].apply(remove_acents)
        arquivo['tipo_acidente'] = arquivo['tipo_acidente'].apply(remove_acents)
        arquivo['classificacao_acidente'] = arquivo['classificacao_acidente'].apply(remove_acents)
        arquivo['dia_semana'] = arquivo['dia_semana'].apply(remove_acents)
        arquivo['condicao_metereologica'] = arquivo['condicao_metereologica'].apply(remove_acents)
        arquivo['tipo_pista'] = arquivo['tipo_pista'].apply(remove_acents)
        arquivo['tracado_via'] = arquivo['tracado_via'].apply(remove_acents)
        arquivo['uso_solo'] = arquivo['uso_solo'].apply(remove_acents)

        arquivo['br'] = pd.to_numeric(arquivo['br'], errors='coerce').astype('Int64')
        arquivo['km'] = pd.to_numeric(arquivo['km'], errors='coerce')

        # Padronização dos dias da semana
        arquivo['dia_semana'] = arquivo['dia_semana'].str.lower().str.strip()

        padronizacao_dia_semana = {
            'segunda' : 'segunda-feira',
            'terca' : 'terca-feira',
            'quarta' : 'quarta-feira',
            'quinta' : 'quinta-feira',
            'sexta' : 'sexta-feira'
        }

        arquivo['dia_semana'] = arquivo['dia_semana'].replace(padronizacao_dia_semana)

    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {e}")

    return arquivo


def filtro(arquivo):
    try:
        # Filtragem do dataframe para o estado do Rio de Janeiro
        arquivo = arquivo[arquivo['uf'] == 'RJ']
        arquivo = arquivo[arquivo['municipio'].isin(['RIO DE JANEIRO', 'NITEROI', 'SAO GONCALO'])]

        # atualização do dataframe para o trecho da BR-101 no estado do Rio de Janeiro, entre os km 299 e 334
        arquivo = arquivo[arquivo['br'] == 101]
        arquivo = arquivo[((arquivo['km'] >= 299) & (arquivo['km'] <= 334))]

    except Exception as e:
        print(f"Erro ao filtrar o arquivo {arquivo}: {e}")

    return arquivo

def regressao_linear(arquivo):
    try:
        df_modelo = arquivo[['fase_dia', 'condicao_metereologica', 'veiculos']].copy()

        X = pd.get_dummies(df_modelo[['fase_dia', 'condicao_metereologica']], drop_first=True)
        y = df_modelo['veiculos']

        modelo = LinearRegression()
        modelo.fit(X, y)

        tabela_coeficientes = pd.DataFrame({
            'Fator_Visibilidade': X.columns,
            'Coeficiente': modelo.coef_
        })

        tabela_coeficientes = tabela_coeficientes.sort_values(by='Coeficiente', ascending=False)

        print("=== IMPACTO DA VISIBILIDADE NO NUMERO DE VEICULOS ===")
        print(tabela_coeficientes)

        print(f"\nMedia base de veiculos por acidente (Condicoes Normais): {modelo.intercept_:.2f}")

        # 1. Tratar os nomes dos rótulos para melhor exibição no gráfico
        tabela_grafico = tabela_coeficientes.copy()
        tabela_grafico['Rotulo'] = (
            tabela_grafico['Fator_Visibilidade']
            .str.replace('condicao_metereologica_', 'Clima: ')
            .str.replace('fase_dia_', 'Fase: ')
        )

        # 2. Definir cores diferenciadas (Verde para aumento, Vermelho/Laranja para redução)
        cores = ['#2ecc71' if c > 0 else '#e74c3c' for c in tabela_grafico['Coeficiente']]

        # 3. Inicializar a figura
        plt.figure(figsize=(10, 6))

        # 4. Desenhar as barras horizontais
        barras = plt.barh(
            tabela_grafico['Rotulo'],
            tabela_grafico['Coeficiente'],
            color=cores,
            edgecolor='black',
            alpha=0.85
        )

        # 5. Adicionar linha de referência no eixo zero
        plt.axvline(0, color='black', linestyle='--', linewidth=1.2)

        # 6. Adicionar os valores exatos na ponta de cada barra
        for barra in barras:
            largura = barra.get_width()
            posicao_x = largura + (0.005 if largura >= 0 else -0.015)
            alinhamento = 'left' if largura >= 0 else 'right'
            plt.text(
                posicao_x,
                barra.get_y() + barra.get_height() / 2,
                f'{largura:+.3f}',
                va='center',
                ha=alinhamento,
                fontsize=9,
                fontweight='bold'
            )

        # 7. Títulos, rótulos e formatação
        plt.title(
            f'Impacto no Número de Veículos por Acidente\n'
            f'(Média Base em Condições Normais: {modelo.intercept_:.2f} veículos)',
            fontsize=12,
            fontweight='bold',
            pad=15
        )
        plt.xlabel('Variação no Número de Veículos (Coeficiente Beta)', fontsize=10)
        plt.ylabel('Fatores Avaliados', fontsize=10)
        plt.grid(axis='x', linestyle=':', alpha=0.6)

        # 8. Ajuste de espaçamento e exibição
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro ao realizar a primeira regressao linear: {e}")

    try:
        arquivo['horario'] = pd.to_datetime(arquivo['horario'], format='%H:%M:%S', errors='coerce').dt.hour
        arquivo['vitimas_graves_mortes'] = arquivo['mortos'] + arquivo['feridos']

        def categorizar_periodo(h):
            if 0 <= h < 6:
                return 'Madrugada'
            elif 6 <= h < 12:
                return 'Manha'
            elif 12 <= h < 18:
                return 'Tarde'
            else:
                return 'Noite'

        arquivo['periodo_dia'] = arquivo['horario'].apply(categorizar_periodo)

        X = pd.get_dummies(arquivo[['periodo_dia']], drop_first=True)
        y = arquivo['vitimas_graves_mortes']

        modelo_2 = LinearRegression()
        modelo_2.fit(X, y)

        # Corrigido para modelo_2
        print(f"\nMedia de Vitimas na Madrugada: {modelo_2.intercept_:.4f}")
        print("\nImpacto dos demais periodos em relacao a Madrugada:")

        # Corrigido para modelo_2
        tabela_coeficientes = pd.DataFrame({
            'Periodo': X.columns.str.replace('periodo_dia_', ''),
            'Diferenca em relacao a Madrugada': modelo_2.coef_
        })

        print(tabela_coeficientes.to_string(index=False))

        # Ajustado para usar 'arquivo' e a coluna 'horario'
        agrupado_hora = arquivo.groupby('horario').agg(
            media_vitimas=('vitimas_graves_mortes', 'mean'),
            total_acidentes=('horario', 'count')
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Volume de Acidentes
        color = 'tab:blue'
        ax1.set_xlabel('Hora do Dia (0h - 23h)')
        ax1.set_ylabel('Total de Acidentes (Volume)', color=color)
        ax1.bar(agrupado_hora['horario'], agrupado_hora['total_acidentes'], color=color, alpha=0.3)
        ax1.set_xticks(range(0, 24))

        # Taxa de Gravidade Calculada
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Média de Vítimas por Acidente', color=color)
        ax2.plot(agrupado_hora['horario'], agrupado_hora['media_vitimas'], color=color, marker='o', linewidth=2)

        plt.title('Regressão Linear (scikit-learn): Severidade por Horário vs. Volume de Sinistros')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    except Exception as e:
        print(f"Erro ao realizar a segunda regressao linear: {e}")

if __name__ == "__main__":
    # Definição dos diretorios principais dos arquivos csv
    prf_bruto = './PRF - Dados Brutos'
    prf_tratado = './PRF - Dados Tratados'

    os.makedirs(prf_tratado, exist_ok=True)

    arquivos_csv_bruto = glob.glob(os.path.join(prf_bruto, '*.csv'))

    if not arquivos_csv_bruto:
        print("Nenhum arquivo CSV encontrado no diretório especificado.")
    else:
        for arquivo in arquivos_csv_bruto:
            print(f"Processando arquivo: {arquivo}")
            df = pd.read_csv(
                arquivo,
                encoding='iso-8859-1',
                sep=';',
                na_values=['', ' ', 'NA', 'N/A', 'na', 'n/a', 'null', 'NULL'],
                decimal=',',
                low_memory=False
            )

            df = limpeza_e_processamento(df)
            df = filtro(df)

            match = re.search(r'\d{4}', arquivo)
            ano = match.group() if match else 'sem_ano'

            nome_saida = f'datatran{ano}_tratado.csv'
            caminho_saida = os.path.join(prf_tratado, nome_saida)

            df.to_csv(
                caminho_saida,
                index=False,
                sep=';',
                encoding='utf-8',
                na_rep=''
            )

    arquivo_csv_tratado = glob.glob(os.path.join(prf_tratado, '*.csv'))

    lista_dfs = []

    if not arquivo_csv_tratado:
      print("Nenhum arquivo tratado encontrado.")
    else:
      for arquivo in arquivo_csv_tratado:
        print(f"Lendo arquivo para juncao: {arquivo}")
        df_ano = pd.read_csv(
            arquivo,
            sep=';',
            encoding='utf-8',
            low_memory=False
        )
        lista_dfs.append(df_ano)

    print("Concatenando todos os anos...")
    # Junta todos os dataframes empilhando-os
    df_consolidado = pd.concat(lista_dfs, ignore_index=True)

    # Salva o arquivo final consolidado
    df_consolidado.to_csv(
        'datatran_consolidado.csv',
        index=False,
        sep=';',
        encoding='utf-8'
    )
    print(f"Total de registros: {len(df_consolidado)}")

    csv_consolidado = pd.read_csv(
        'datatran_consolidado.csv',
        sep=';',
        encoding='utf-8',
        low_memory=False
    )

    regressao_linear(csv_consolidado)