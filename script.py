import pandas as pd
import unicodedata as uni
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

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
                'sentido_via', 'ano', 'feridos_leves', 'ilesos', 'ignorados', 'feridos'
            ], errors='ignore')
            
            if arquivo['km'].dtype != 'float64':
                arquivo['km'] = arquivo['km'].str.replace(',', '.')
                
        else:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'ano', 'feridos_leves', 'ilesos', 'ignorados', 'feridos', 'latitude', 'longitude', 'regional', 'delegacia', 'uop'
            ], errors='ignore')
        
        # Correção dos dados para padronização do dataframe
        arquivo['br'] = arquivo['br'].replace('(null)', pd.NA)
        arquivo['km'] = arquivo['km'].replace('(null)', pd.NA)
        arquivo['causa_acidente'] = arquivo['causa_acidente'].apply(remove_acents)
        arquivo['tipo_acidente'] = arquivo['tipo_acidente'].apply(remove_acents)
        arquivo['classificacao_acidente'] = arquivo['classificacao_acidente'].apply(remove_acents)
        arquivo['dia_semana'] = arquivo['dia_semana'].apply(remove_acents)
        arquivo['condicao_metereologica'] = arquivo['condicao_metereologica'].apply(remove_acents)
        arquivo['tipo_pista'] = arquivo['tipo_pista'].apply(remove_acents)
        arquivo['tracado_via'] = arquivo['tracado_via'].apply(remove_acents)
        arquivo['uso_solo'] = arquivo['uso_solo'].apply(remove_acents)
        
        # definição dos tipos de dados das colunas
        if arquivo['br'].dtype != 'Int64' and arquivo['km'].dtype != 'float64':
            arquivo = arquivo.astype({
                'br': 'Int64',
                'km': 'float64'
            })
        
            
    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {e}")
        
    return arquivo


def filtro(arquivo):
    try:
        # Filtragem do dataframe para o estado do Rio de Janeiro
        arquivo = arquivo[arquivo['uf'] == 'RJ']
        arquivo = arquivo[arquivo['municipio'] != 'ITABORAI']

        # atualização do dataframe para o trecho da BR-101 no estado do Rio de Janeiro, entre os km 299 e 334
        arquivo = arquivo[arquivo['br'] == 101]
        arquivo = arquivo[((arquivo['km'] >= 299) & (arquivo['km'] <= 334))]
        
    except Exception as e:
        print(f"Erro ao filtrar o arquivo {arquivo}: {e}")
        
    return arquivo

def regressao_linear(arquivo, nome_arquivo):
    try:
        # Ajustes para a regressão linear
        arquivo['data_inversa'] = pd.to_datetime(arquivo['data_inversa'], format='mixed', dayfirst=True)
        arquivo['mes'] = arquivo['data_inversa'].dt.month
        sazonalidade_mensal = arquivo.groupby('mes').size().reset_index(name='total_acidentes')
        
        
        match = re.search(r'\d{4}', nome_arquivo)
        
        X_meses = sazonalidade_mensal['mes'].to_numpy()
        y_acidentes = sazonalidade_mensal['total_acidentes'].to_numpy()
        
        coeficientes_mes = np.polyfit(X_meses, y_acidentes, 1)
        reta_tendencia_mes = np.poly1d(coeficientes_mes)        
        
        plt.figure(figsize=(10, 5))
        
        
        plt.plot(X_meses, y_acidentes, marker='o', color='blue', linewidth=2, label='Acidentes Reais (Sazonalidade)')
        plt.plot(X_meses, reta_tendencia_mes(X_meses), color='red', linestyle='--', label=f'Tendência Linear (y = {coeficientes_mes[0]:.2f}x + {coeficientes_mes[1]:.0f})')
        
        meses_labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        plt.xticks(range(1, 13), meses_labels)
        plt.xlim(1, 12) # Garante que o gráfico sempre mostre de Jan a Dez
        plt.title(f'Sazonalidade Temporal: Acidentes ao Longo dos Meses do ano de {match.group()}', fontsize=14)
        plt.xlabel('Mês do Ano')
        plt.ylabel('Volume de Acidentes')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.show()
        
        df_br101 = df[df['br'] == 101].copy()
        
        df_br101['km_arredondado'] = np.floor(df_br101['km'])
        densidade_espacial = df_br101.groupby('km_arredondado').size().reset_index(name='total_acidentes')
        
        X_km = densidade_espacial['km_arredondado'].to_numpy()
        y_km_acidentes = densidade_espacial['total_acidentes'].to_numpy()
        
        coeficientes_km = np.polyfit(X_km, y_km_acidentes, 1)
        reta_tendencia_km = np.poly1d(coeficientes_km)
        
        plt.figure(figsize=(12, 5))
        plt.scatter(X_km, y_km_acidentes, color='orange', alpha=0.5, label='Acidentes por KM')
        plt.plot(X_km, reta_tendencia_km(X_km), color='darkred', linewidth=2, label='Tendência de Agravamento Espacial')
        
        plt.title(f'Distribuição Espacial: Frequência de Acidentes ao Longo da BR-101 no ano de {match.group()}', fontsize=14)
        plt.xlabel('Quilômetro (KM) da Rodovia')
        plt.ylabel('Quantidade de Acidentes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Erro ao calcular a regressão linear: {e}")

if __name__ == "__main__":
    # Definição dos diretorios principais dos arquivos csv
    prf_bruto = './PRF - dados brutos/'
    prf_tratado = './PRF - Dados Tratados/'

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
            
            df.to_csv(
                f'./PRF - Dados Tratados/datatran{match.group()}_tratado.csv',
                index=False,
                sep=';',
                encoding='utf-8',
                na_rep=''
            )
    
    arquivo_csv_tratado = glob.glob(os.path.join(prf_tratado, '*.csv'))
    
    if not arquivo_csv_tratado:
        print("Nenhum arquivo CSV tratado encontrado no diretório especificado.")
    else:
        for arquivo in arquivo_csv_tratado:
            print(f"Processando arquivo tratado: {arquivo}")
            df = pd.read_csv(
                arquivo,
                encoding='utf-8',
                sep=';',
                na_values=['', ' ', 'NA', 'N/A', 'na', 'n/a', 'null', 'NULL'],
                low_memory=False
            )
            
            df = regressao_linear(df, arquivo)
            
            
            
            
