"""
ETL — Etapa T (Transform), Fase 1: triagem binária local
Classifica cada comentário só como "nocivo" ou "nenhum", sem distinguir
categoria ainda. Como junta todas as categorias de ódio do ToLD-Br em
uma única classe, o volume de treino por classe fica muito maior —
resolve o problema de categorias raras (racismo, xenofobia) que
travava o classificador multiclasse.

Roda 100% local — sem API, sem custo, sem rate limit.
"""

import csv
import os
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TOLD_BR_FILE = "data/raw/told_br/ToLD-BR.csv"
INPUT_FILE = "data/raw/comentarios_para_analise.csv"
OUTPUT_FILE = "data/processed/comentarios_binario.csv"

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# colunas do ToLD-Br que contam como "nocivo" se score >= 1
COLUNAS_ODIO = ["homophobia", "racism", "misogyny", "xenophobia", "obscene", "insult"]

LIMIAR_CONFIANCA = 0.55


def carregar_told_br_binario(path: str) -> tuple[list[str], list[str]]:
    textos, rotulos = [], []

    with open(path, encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        for row in leitor:
            texto = row.get("text", "").strip()
            if not texto:
                continue

            eh_nocivo = False
            for col in COLUNAS_ODIO:
                try:
                    valor = float(row.get(col, 0) or 0)
                except ValueError:
                    valor = 0
                if valor >= 1:
                    eh_nocivo = True
                    break

            textos.append(texto)
            rotulos.append("nocivo" if eh_nocivo else "nenhum")

    return textos, rotulos


def treinar_classificador_binario(textos: list[str], rotulos: list[str], encoder: SentenceTransformer):
    print("Distribuição binária no dado de treino (ToLD-Br):")
    for rot, qtd in Counter(rotulos).most_common():
        print(f"  {rot}: {qtd}")

    print(f"\nGerando embeddings de {len(textos)} exemplos de treino...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, rotulos, test_size=0.15, random_state=42, stratify=rotulos
    )

    print("Treinando classificador binário (Regressão Logística)...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("\n=== Avaliação (binário, sem limiar) ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n=== Avaliação (binário, com limiar de confiança) ===")
    probs_test = clf.predict_proba(X_test)
    classes = clf.classes_
    y_pred_limiar = []
    for probs in probs_test:
        idx = int(np.argmax(probs))
        rotulo = classes[idx]
        if rotulo == "nocivo" and probs[idx] < LIMIAR_CONFIANCA:
            rotulo = "nenhum"
        y_pred_limiar.append(rotulo)
    print(classification_report(y_test, y_pred_limiar, zero_division=0))

    return clf


def classificar_binario(input_file: str, output_file: str, clf, encoder: SentenceTransformer) -> None:
    with open(input_file, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    textos = [row.get("text", "").strip() for row in comentarios]
    print(f"\nGerando embeddings de {len(textos)} comentários coletados...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    print("Classificando (binário)...")
    probabilidades = clf.predict_proba(X)
    classes = clf.classes_

    qtd_nocivos = 0
    for row, probs in zip(comentarios, probabilidades):
        idx = int(np.argmax(probs))
        rotulo = classes[idx]
        confianca_valor = probs[idx]

        if rotulo == "nocivo" and confianca_valor < LIMIAR_CONFIANCA:
            rotulo = "nenhum"

        if rotulo == "nocivo":
            qtd_nocivos += 1

        row["nocivo_binario"] = rotulo
        row["confianca_binaria"] = f"{confianca_valor:.2f}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comentarios[0].keys())
        writer.writeheader()
        writer.writerows(comentarios)

    print(f"\nFase 1 concluída: {output_file}")
    print(f"Total processado: {len(comentarios)}")
    print(f"Marcados como nocivos (vão para a Fase 2, categorização via LLM): {qtd_nocivos}")


if __name__ == "__main__":
    print(f"Carregando modelo de embeddings: {MODELO_EMBEDDING}...")
    encoder = SentenceTransformer(MODELO_EMBEDDING)

    textos_treino, rotulos_treino = carregar_told_br_binario(TOLD_BR_FILE)
    clf = treinar_classificador_binario(textos_treino, rotulos_treino, encoder)

    classificar_binario(INPUT_FILE, OUTPUT_FILE, clf, encoder)