import pandas as pd

df = pd.read_csv(
        './PRF - Dados Brutos/datatran2016.csv',
        
        encoding='iso-8859-1',
        sep=';',
        na_values=['', ' ', 'NA', 'N/A', 'na', 'n/a', 'null', 'NULL'],
        decimal=',',
        low_memory=False
    )

df = df.drop(columns=[
                'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'ano', 'pessoas',
                'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos', 'mortos', 'fase_dia'
            ], errors='ignore')

acidente = df[df['id'] == 83529886]
if not acidente.empty:
    print(acidente)
else:
    print("Acidente com o ID 96363 não foi encontrado na base.")
# print(acidente_especifico)