import pandas as pd

# Carrega o arquivo gerado
df = pd.read_json('comentarios.json', lines=True)

# Exibe as primeiras linhas e o total de comentários
print(f"Total de comentários extraídos: {len(df)}")
print(df[['author', 'text', 'votes']].head())

# Se quiser salvar em Excel ou CSV para analisar mais fácil:
df.to_csv('comentarios_para_analise.csv', index=False, encoding='utf-8-sig')