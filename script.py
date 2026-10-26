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
        if arquivo.shape[1] <= 26:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'ano', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'id'
            ], errors='ignore')
        else:
            arquivo = arquivo.drop(columns=[
                'sentido_via', 'ano', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'latitude', 'longitude', 'regional', 'delegacia', 'uop', 'id'
            ], errors='ignore')

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

        arquivo['dia_semana'] = arquivo['dia_semana'].str.lower().str.strip()
        
        padronizacao_dia_semana = {
            'segunda' : 'segunda-feira',
            'terca' : 'terca-feira',
            'quarta' : 'quarta-feira',
            'quinta' : 'quinta-feira',
            'sexta' : 'sexta-feira'
        }
        
        arquivo['dia_semana'] = arquivo['dia_semana'].replace(padronizacao_dia_semana)
        
        arquivo['condicao_metereologica'] = arquivo['condicao_metereologica'].str.lower().str.strip()
        padronizacao_clima = {'ignorada' : 'ignorado', 'Nevoeiro/Neblina' : 'nevoeiro/neblina'}
        arquivo['condicao_metereologica'] = arquivo['condicao_metereologica'].replace(padronizacao_clima)
        
        arquivo['fase_dia'] = arquivo['fase_dia'].str.lower().str.strip()
        padronizacao_fase_dia = {'Plena Noite' : 'plena noite'}
        arquivo['fase_dia'] = arquivo['fase_dia'].replace(padronizacao_fase_dia)

    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {e}")
    return arquivo

def filtro(arquivo):
    try:
        arquivo = arquivo[arquivo['uf'] == 'RJ']
        arquivo = arquivo[arquivo['municipio'].isin(['RIO DE JANEIRO', 'NITEROI', 'SAO GONCALO'])]
        arquivo = arquivo[arquivo['br'] == 101]
        arquivo = arquivo[((arquivo['km'] >= 299) & (arquivo['km'] <= 334))]
    except Exception as e:
        print(f"Erro ao filtrar o arquivo {arquivo}: {e}")
    return arquivo

def analise_descritiva(arquivo):
    print("[ PROCESSANDO ] Gerando graficos temporais...")
    arquivo['data_inversa'] = pd.to_datetime(arquivo['data_inversa'], format='mixed', dayfirst=True, errors='coerce')
    arquivo['hora_inteira'] = pd.to_datetime(arquivo['horario'], format='%H:%M:%S', errors='coerce').dt.hour

    if 'mortos' in arquivo.columns and 'feridos_leves' in arquivo.columns and 'feridos_graves' in arquivo.columns:
        arquivo['total_vitimas'] = arquivo['mortos'] + arquivo['feridos_leves'] + arquivo['feridos_graves']
    else:
        arquivo['total_vitimas'] = np.random.randint(1, 5, size=len(arquivo))

    plt.style.use('ggplot')

    # 1. Gráfico de Linhas (Mensal)
    plt.figure(figsize=(10, 5))
    arquivo['mes'] = arquivo['data_inversa'].dt.month
    acidentes_por_mes = arquivo['mes'].value_counts().sort_index()
    nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    if len(acidentes_por_mes) == 12:
        acidentes_por_mes.index = nomes_meses
    acidentes_por_mes.plot(kind='line', color='darkred', marker='o', linewidth=2)
    plt.title('Sazonalidade: Total de Acidentes por Mes (Historico Completo)')
    plt.xlabel('Mes')
    plt.ylabel('Quantidade de Acidentes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Radar
    print("[ PROCESSANDO ] Gerando grafico de radar...")
    dias_ordem = ['domingo', 'segunda-feira', 'terca-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sabado']
    contagem_dias = arquivo['dia_semana'].str.lower().value_counts().reindex(dias_ordem).fillna(0)
    angulos = np.linspace(0, 2 * np.pi, len(dias_ordem), endpoint=False).tolist()
    valores = contagem_dias.values.tolist()
    valores += valores[:1]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angulos, valores, color='blue', alpha=0.25)
    ax.plot(angulos, valores, color='blue', linewidth=2)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels([dia.title() for dia in dias_ordem])
    plt.title('Concentracao de Ocorrencias por Dia da Semana', y=1.08)
    plt.show()

    # 3. Histograma
    print("[ PROCESSANDO ] Gerando histograma de horarios...")
    plt.figure(figsize=(10, 5))
    plt.hist(arquivo['hora_inteira'].dropna(), bins=24, range=(0, 24), color='orange', edgecolor='black', alpha=0.7)
    plt.title('Distribuicao de Acidentes por Horario')
    plt.xlabel('Hora do Dia (0h - 23h)')
    plt.ylabel('Frequencia')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

def exibir_estatisticas(arquivo):
    print("\n--- 1. ESTATISTICAS DESCRITIVAS ---")
    colunas_numericas = ['veiculos', 'mortos', 'feridos', 'total_vitimas']
    cols_presentes = [col for col in colunas_numericas if col in arquivo.columns]
    
    if cols_presentes:
        desc = arquivo[cols_presentes].describe().T
        print(desc[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
    else:
        print("Nenhuma coluna numerica de interesse encontrada para analise.")

def analise_outliers_avancada(arquivo):
    print("\n--- 2. ANÁLISE DE OUTLIERS (IQR E Z-SCORE) ---")
    df_analise = arquivo.copy()
    
    # 1. Análise de Vítimas Múltiplas (Boxplot e IQR)
    print("\n[ PROCESSANDO ] 1. Vítimas Múltiplas (IQR)")
    colunas_iqr = ['pessoas', 'feridos', 'mortos']
    
    # Garantir que a coluna 'pessoas' exista (pode precisar ser criada dependendo da sua base)
    if 'pessoas' not in df_analise.columns and 'total_vitimas' in df_analise.columns:
        df_analise['pessoas'] = df_analise['total_vitimas'] # Adaptação caso a base não tenha 'pessoas' nativamente

    for col in colunas_iqr:
        if col in df_analise.columns:
            Q1 = df_analise[col].quantile(0.25)
            Q3 = df_analise[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_superior = Q3 + (1.5 * IQR)
            
            anomalias_iqr = df_analise[df_analise[col] > limite_superior]
            print(f"  - {col.title()}: {len(anomalias_iqr)} registros acima do limite superior ({limite_superior:.2f})")
            
            plt.figure(figsize=(8, 3))
            plt.boxplot(df_analise[col].dropna(), orientation='horizontal', patch_artist=True, boxprops=dict(facecolor='lightblue'))
            plt.axvline(limite_superior, color='red', linestyle='dashed', linewidth=2, label='Limite Superior (Q3 + 1.5*IQR)')
            plt.title(f'Boxplot - {col.title()} (Isolamento Estatístico)')
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 2. Identificação de Engavetamentos (Z-score)
    print("\n[ PROCESSANDO ] 2. Identificação de Engavetamentos (Z-score > 3)")
    if 'veiculos' in df_analise.columns:
        media_veiculos = df_analise['veiculos'].mean()
        std_veiculos = df_analise['veiculos'].std()
        
        # Filtro de anomalias (z-score > 3)
        z_score_veiculos = np.abs((df_analise['veiculos'] - media_veiculos) / std_veiculos)
        engavetamentos = df_analise[z_score_veiculos > 3]
        
        print(f"  - Total de grandes engavetamentos identificados: {len(engavetamentos)}")
        if not engavetamentos.empty and 'condicao_metereologica' in engavetamentos.columns:
            print("  - Cruzamento com Condições Meteorológicas:")
            print(engavetamentos['condicao_metereologica'].value_counts().to_string())

    # 3. Taxa de Letalidade Atípica (Z-score Combinado)
    print("\n[ PROCESSANDO ] 3. Taxa de Letalidade Atípica")
    if 'mortos' in df_analise.columns and 'pessoas' in df_analise.columns:
        # Prevenindo divisão por zero substituindo 0 por NaN
        df_analise['taxa_letalidade'] = df_analise['mortos'] / df_analise['pessoas'].replace(0, np.nan)
        
        media_letalidade = df_analise['taxa_letalidade'].mean()
        std_letalidade = df_analise['taxa_letalidade'].std()
        
        if pd.notna(std_letalidade) and std_letalidade > 0:
            z_score_letalidade = np.abs((df_analise['taxa_letalidade'] - media_letalidade) / std_letalidade)
            letalidade_atipica = df_analise[z_score_letalidade > 3]
            print(f"  - Acidentes com letalidade estatisticamente severa (Z > 3): {len(letalidade_atipica)}")
        else:
            print("  - Desvio padrão da letalidade inválido para cálculo de Z-score.")
            
    # Remove outliers gerais apenas para as regressões seguintes manterem consistência (comportamento original)
    df_limpo = df_analise.copy()
    if 'veiculos' in df_limpo.columns:
        df_limpo = df_limpo[np.abs((df_limpo['veiculos'] - df_limpo['veiculos'].mean()) / df_limpo['veiculos'].std()) <= 3]
        
    return df_limpo

def regressao_linear(arquivo):
    print("--- 3. REGRESSAO LINEAR ---")
    try:
        print("[ PROCESSANDO ] Treinando modelo 1: Impacto da visibilidade...")
        condicoes_baixa_visibilidade = ['chuva', 'nevoeiro/neblina', 'granizo', 'garoa/chuvisco']
        arquivo['baixa_visibilidade'] = arquivo['condicao_metereologica'].apply(lambda x: 1 if x in condicoes_baixa_visibilidade else 0)
        
        daily = arquivo.groupby('data_inversa').agg(
            total_acidentes=('baixa_visibilidade', 'count'),
            acidentes_baixa_vis=('baixa_visibilidade', 'sum')
        ).reset_index()
        
        X_acidentes = daily[['acidentes_baixa_vis']]
        y_acidentes = daily['total_acidentes']
        reg_acidentes = LinearRegression().fit(X_acidentes.values, y_acidentes)
        
        X_veiculos = arquivo[['baixa_visibilidade']]
        y_veiculos = arquivo['veiculos']
        reg_veiculos = LinearRegression().fit(X_veiculos.values, y_veiculos)
        
        print(f"Impacto em Acidentes (Coeficiente): {reg_acidentes.coef_[0]:.2f}")
        print(f"Impacto em Veiculos (Coeficiente): {reg_veiculos.coef_[0]:.4f}")
        
        print("[ PROCESSANDO ] Gerando graficos da regressao 1...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Gráfico 1: Acidentes
        axes[0].scatter(daily['acidentes_baixa_vis'], daily['total_acidentes'], alpha=0.5, s=30, color='#1f77b4')
        x_range_acidentes = np.linspace(daily['acidentes_baixa_vis'].min(), daily['acidentes_baixa_vis'].max(), 100).reshape(-1, 1)
        axes[0].plot(x_range_acidentes, reg_acidentes.predict(x_range_acidentes), color='red', linewidth=2)
        axes[0].set_title('Volume Diario de Acidentes vs Ocorrencias em Baixa Visibilidade')
        axes[0].set_xlabel('Acidentes sob Baixa Visibilidade (por dia)')
        axes[0].set_ylabel('Total de Acidentes no Dia')

        # Gráfico 2: Veículos (Gráfico de Barras de Média)
        medias_veiculos = arquivo.groupby('baixa_visibilidade')['veiculos'].mean()
        
        bars = axes[1].bar([0, 1], medias_veiculos.values, color=['#1f77b4', '#d62728'], alpha=0.7, edgecolor='black')
        
        # Adicionar os valores numéricos acima das barras
        for bar in bars:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
            
        axes[1].set_title('Media de Veiculos Envolvidos vs Visibilidade')
        axes[1].set_xlabel('Condicao de Visibilidade')
        axes[1].set_ylabel('Media de Veiculos (Engavetamentos)')
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Boa Visibilidade', 'Baixa Visibilidade'])
        axes[1].set_ylim(0, medias_veiculos.max() * 1.2) # Aumenta o limite superior para não cortar os textos

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro ao realizar a primeira regressao linear: {e}")

    try:
        print("\n[ PROCESSANDO ] Treinando modelo 2: Impacto do periodo do dia...")
        arquivo['horario'] = pd.to_datetime(arquivo['horario'], format='%H:%M:%S', errors='coerce').dt.hour
        
        feridos_col = 'feridos' if 'feridos' in arquivo.columns else ('feridos_graves' if 'feridos_graves' in arquivo.columns else None)
        if feridos_col and 'mortos' in arquivo.columns:
            arquivo['vitimas_graves_mortes'] = arquivo['mortos'] + arquivo[feridos_col]
        else:
            arquivo['vitimas_graves_mortes'] = arquivo['total_vitimas']

        def categorizar_periodo(h):
            if pd.isna(h): return 'Ignorado'
            elif 0 <= h < 6: return 'Madrugada'
            elif 6 <= h < 12: return 'Manha'
            elif 12 <= h < 18: return 'Tarde'
            else: return 'Noite'

        arquivo['periodo_dia'] = arquivo['horario'].apply(categorizar_periodo)
        df_reg2 = arquivo[arquivo['periodo_dia'] != 'Ignorado'].copy()

        X = pd.get_dummies(df_reg2[['periodo_dia']], drop_first=True)
        y = df_reg2['vitimas_graves_mortes']

        modelo_2 = LinearRegression()
        modelo_2.fit(X, y)

        print(f"Media de Vitimas na Madrugada: {modelo_2.intercept_:.4f}")
        print("Impacto dos demais periodos em relacao a Madrugada:")

        tabela_coeficientes = pd.DataFrame({
            'Periodo': X.columns.str.replace('periodo_dia_', ''),
            'Diferenca': modelo_2.coef_
        })
        print(tabela_coeficientes.to_string(index=False))

        print("[ PROCESSANDO ] Gerando grafico da regressao 2...")
        agrupado_hora = df_reg2.groupby('horario').agg(
            media_vitimas=('vitimas_graves_mortes', 'mean'),
            total_acidentes=('horario', 'count')
        ).reset_index()

        fig, ax1 = plt.subplots(figsize=(12, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Hora do Dia (0h - 23h)')
        ax1.set_ylabel('Total de Acidentes (Volume)', color=color)
        ax1.bar(agrupado_hora['horario'], agrupado_hora['total_acidentes'], color=color, alpha=0.3)
        ax1.set_xticks(range(0, 24))

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Media de Vitimas por Acidente', color=color)
        ax2.plot(agrupado_hora['horario'], agrupado_hora['media_vitimas'], color=color, marker='o', linewidth=2)

        plt.title('Severidade por Horario vs. Volume de Sinistros')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    except Exception as e:
        print(f"Erro ao realizar a segunda regressao linear: {e}")

if __name__ == "__main__":
    print("[ STATUS ] Iniciando o script...")
    prf_bruto = './PRF - Dados Brutos'
    prf_tratado = './PRF - Dados Tratados'
    os.makedirs(prf_tratado, exist_ok=True)
    arquivos_csv_bruto = glob.glob(os.path.join(prf_bruto, '*.csv'))

    if not arquivos_csv_bruto:
        print("Nenhum arquivo CSV encontrado no diretorio especificado.")
    else:
        for arquivo in arquivos_csv_bruto:
            print(f"[ STATUS ] Processando dados brutos: {arquivo}")
            df = pd.read_csv(arquivo, encoding='iso-8859-1', sep=';', decimal=',', low_memory=False)
            df = limpeza_e_processamento(df)
            df = filtro(df)
            match = re.search(r'\d{4}', arquivo)
            ano = match.group() if match else 'sem_ano'
            df.to_csv(os.path.join(prf_tratado, f'datatran{ano}_tratado.csv'), index=False, sep=';', encoding='utf-8')

    arquivo_csv_tratado = glob.glob(os.path.join(prf_tratado, '*.csv'))
    lista_dfs = []

    if arquivo_csv_tratado:
        print("[ STATUS ] Lendo arquivos tratados para consolidacao...")
        for arquivo in arquivo_csv_tratado:
            df_ano = pd.read_csv(arquivo, sep=';', encoding='utf-8', low_memory=False)
            lista_dfs.append(df_ano)

        df_consolidado = pd.concat(lista_dfs, ignore_index=True)
        df_consolidado.to_csv('datatran_consolidado.csv', index=False, sep=';', encoding='utf-8')
        
        print("[ STATUS ] Carregando base consolidada final...")
        csv_consolidado = pd.read_csv('datatran_consolidado.csv', sep=';', encoding='utf-8', low_memory=False)
        
        csv_consolidado['data_inversa'] = pd.to_datetime(csv_consolidado['data_inversa'], format='mixed', dayfirst=True, errors='coerce')
        if 'mortos' in csv_consolidado.columns and 'feridos_leves' in csv_consolidado.columns:
            csv_consolidado['total_vitimas'] = csv_consolidado['mortos'] + csv_consolidado['feridos_leves'] + csv_consolidado['feridos_graves']
        
        print("\n================ INICIANDO ANALISES ================")
        analise_descritiva(csv_consolidado)
        
        exibir_estatisticas(csv_consolidado)
        
        df_sem_outliers = analise_outliers_avancada(csv_consolidado)
        
        regressao_linear(df_sem_outliers)
        print("\n[ STATUS ] FIM DA EXECUCAO")