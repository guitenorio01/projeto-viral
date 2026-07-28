"""
ETL — Etapa T (Transform), versão embeddings (sem API externa)
Treina um classificador leve (sentence embeddings + Regressão Logística)
usando o ToLD-Br como dado rotulado, e aplica nos comentários coletados.
Roda 100% local — sem rate limit, sem custo, sem dependência de internet
após o download inicial do modelo.
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TOLD_BR_FILE = "data/raw/told_br/ToLD-BR.csv"
INPUT_FILE = "data/raw/comentarios_roda_viva_erika_hilton_20260724.csv"
OUTPUT_FILE = "data/processed/comentarios_classificados.csv"

# modelo multilingual leve, roda bem em CPU
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# mapeia as colunas do ToLD-Br pras categorias do projeto VIRAL
MAPA_CATEGORIAS = {
    "homophobia": "homofobia_transfobia",
    "racism": "racismo",
    "misogyny": "misoginia",
    "xenophobia": "xenofobia",
    "obscene": "outros_odio",
    "insult": "outros_odio",
}

# só considera exemplo de treino como "nocivo" se pelo menos 2 dos 3
# anotadores concordaram (score >= 2) — reduz ruído de casos duvidosos
SCORE_MINIMO = 2


def carregar_told_br(path: str) -> tuple[list[str], list[str]]:
    """Lê o ToLD-Br e retorna (textos, categoria_dominante) para treino."""
    textos, categorias = [], []

    with open(path, encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        for row in leitor:
            texto = row.get("text", "").strip()
            if not texto:
                continue

            # pega a categoria com maior score entre as colunas de ódio
            scores = {}
            for col_told, categoria_viral in MAPA_CATEGORIAS.items():
                try:
                    valor = float(row.get(col_told, 0) or 0)
                except ValueError:
                    valor = 0
                scores[categoria_viral] = max(scores.get(categoria_viral, 0), valor)

            categoria_final = max(scores, key=scores.get) if scores else "nenhum"
            if scores.get(categoria_final, 0) < SCORE_MINIMO:
                categoria_final = "nenhum"

            textos.append(texto)
            categorias.append(categoria_final)

    return textos, categorias


def treinar_classificador(textos: list[str], categorias: list[str], encoder: SentenceTransformer):
    print(f"Gerando embeddings de {len(textos)} exemplos de treino (ToLD-Br)...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, categorias, test_size=0.15, random_state=42, stratify=categorias
    )

    print("Treinando classificador (Regressão Logística)...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("\n=== Avaliação no conjunto de teste (ToLD-Br) ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    return clf


def classificar_comentarios(input_file: str, output_file: str, clf, encoder: SentenceTransformer) -> None:
    with open(input_file, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    textos = [row.get("text", "").strip() for row in comentarios]
    print(f"\nGerando embeddings de {len(textos)} comentários coletados...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    print("Classificando...")
    predicoes = clf.predict(X)
    probabilidades = clf.predict_proba(X)
    classes = clf.classes_

    # exige confiança mínima pra marcar como nocivo — abaixo disso,
    # o comentário fica como "nenhum" por segurança (evita falso positivo)
    LIMIAR_CONFIANCA = 0.55

    for row, probs in zip(comentarios, probabilidades):
        indice_max = int(np.argmax(probs))
        categoria_prevista = classes[indice_max]
        confianca_valor = probs[indice_max]

        if categoria_prevista != "nenhum" and confianca_valor < LIMIAR_CONFIANCA:
            categoria_final = "nenhum"
        else:
            categoria_final = categoria_prevista

        confianca = "alta" if confianca_valor > 0.7 else "media" if confianca_valor > 0.55 else "baixa"

        row["categoria"] = categoria_final
        row["confianca"] = confianca
        row["justificativa"] = f"classificador embeddings (prob={confianca_valor:.2f})"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comentarios[0].keys())
        writer.writeheader()
        writer.writerows(comentarios)

    print(f"\nTransform concluído: {output_file}")
    print(f"Total processado: {len(comentarios)}")


if __name__ == "__main__":
    print(f"Carregando modelo de embeddings: {MODELO_EMBEDDING}...")
    encoder = SentenceTransformer(MODELO_EMBEDDING)

    textos_treino, categorias_treino = carregar_told_br(TOLD_BR_FILE)
    clf = treinar_classificador(textos_treino, categorias_treino, encoder)

    classificar_comentarios(INPUT_FILE, OUTPUT_FILE, clf, encoder)