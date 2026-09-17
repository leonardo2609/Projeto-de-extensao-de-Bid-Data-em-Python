import pandas as pd
import unicodedata as uni
import os
import glob
import re

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
                'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'ano', 'pessoas',
                'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos', 'mortos'
            ], errors='ignore')
            
            if arquivo['km'].dtype != 'float64':
                arquivo['km'] = arquivo['km'].str.replace(',', '.')
                
        else:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'ano', 'pessoas', 'feridos_leves', 'feridos_graves',
                'ilesos', 'ignorados', 'feridos', 'veiculos', 'mortos', 'latitude', 'longitude', 'regional', 'delegacia', 'uop'
            ], errors='ignore')
        
        # Correção dos dados para padronização do dataframe
        arquivo['br'] = arquivo['br'].replace('(null)', pd.NA)
        arquivo['km'] = arquivo['km'].replace('(null)', pd.NA)
        arquivo['causa_acidente'] = arquivo['causa_acidente'].apply(remove_acents)
        arquivo['tipo_acidente'] = arquivo['tipo_acidente'].apply(remove_acents)
        arquivo['classificacao_acidente'] = arquivo['classificacao_acidente'].apply(remove_acents)
        arquivo['dia_semana'] = arquivo['dia_semana'].apply(remove_acents)
        
        # definição dos tipos de dados das colunas
        if arquivo['br'].dtype != 'Int64' and arquivo['km'].dtype != 'float64':
            arquivo = arquivo.astype({
                'br': 'Int64',
                'km': 'float64'
            })
        
        print(arquivo.info())
            
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

if __name__ == "__main__":
    # Definição dos diretorios principais dos arquivos csv
    prf_bruto = './PRF - dados brutos/'
    prf_tratado = './PRF - Dados Tratados/'

    os.makedirs(prf_tratado, exist_ok=True)

    arquivos_csv = glob.glob(os.path.join(prf_bruto, '*.csv'))
    
    if not arquivos_csv:
        print("Nenhum arquivo CSV encontrado no diretório especificado.")
    else:
        for arquivo in arquivos_csv:
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
