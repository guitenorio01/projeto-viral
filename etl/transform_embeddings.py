"""
ETL — Etapa T (Transform), versão embeddings - FASE 1 (Binária)
Treina um classificador binário leve (Sentence Embeddings + Regressão Logística)
usando o ToLD-Br simplificado em: 1 (Nocivo) vs 0 (Não-Nocivo).

Objetivo: Fazer o filtro inicial pesado de alta performance para extrair
apenas os comentários nocivos que serão categorizados na Fase 2.
"""

import csv
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TOLD_BR_FILE = "data/raw/told_br/ToLD-BR.csv"
INPUT_FILE = "data/raw/comentarios_para_analise.csv"
OUTPUT_FILE = "data/processed/comentarios_classificados.csv"

# Modelo multilingual leve para CPU
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# Limiar de probabilidade mínima para confirmar como 'nocivo' (evita falsos positivos)
LIMIAR_CONFIANCA = 0.55

# Colunas do ToLD-Br que representam qualquer forma de conteúdo nocivo
COLUNAS_ODIO = ["homophobia", "racism", "misogyny", "xenophobia", "obscene", "insult"]

# Concordância mínima de anotadores no ToLD-Br para considerar como nocivo
SCORE_MINIMO = 1


def carregar_told_br_binario(path: str) -> tuple[list[str], list[str]]:
    """Lê o ToLD-Br e mapeia para 2 classes: 'nocivo' vs 'nao_nocivo'."""
    textos, categorias = [], []

    with open(path, encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        for row in leitor:
            texto = row.get("text", "").strip()
            if not texto:
                continue

            # Verifica o maior score em qualquer uma das categorias de ódio
            max_score = 0.0
            for col in COLUNAS_ODIO:
                try:
                    valor = float(row.get(col, 0) or 0)
                    if valor > max_score:
                        max_score = valor
                except ValueError:
                    continue

            # Se atingir o score mínimo em QUALQUER categoria, vira 'nocivo'
            categoria_final = "nocivo" if max_score >= SCORE_MINIMO else "nao_nocivo"

            textos.append(texto)
            categorias.append(categoria_final)

    return textos, categorias


def treinar_classificador_binario(textos: list[str], categorias: list[str], encoder: SentenceTransformer):
    from collections import Counter
    print("=== DISTRIBUIÇÃO DAS CLASSES NO TREINO (ToLD-Br Binário) ===")
    for cat, qtd in Counter(categorias).most_common():
        pct = (qtd / len(categorias)) * 100
        print(f"  {cat}: {qtd} ({pct:.1f}%)")

    print(f"\nGerando embeddings de {len(textos)} exemplos de treino...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, categorias, test_size=0.15, random_state=42, stratify=categorias
    )

    print("\nTreinando classificador binário (Regressão Logística)...")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("\n=== AVALIAÇÃO BINÁRIA - Predict Padrão ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\n=== AVALIAÇÃO BINÁRIA - Com Limiar de Confiança >= {LIMIAR_CONFIANCA} ===")
    probs_test = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    idx_nocivo = classes.index("nocivo")

    y_pred_limiar = []
    for probs in probs_test:
        prob_nocivo = probs[idx_nocivo]
        if prob_nocivo >= LIMIAR_CONFIANCA:
            y_pred_limiar.append("nocivo")
        else:
            y_pred_limiar.append("nao_nocivo")

    print(classification_report(y_test, y_pred_limiar, zero_division=0))

    return clf


def classificar_comentarios(input_file: str, output_file: str, clf, encoder: SentenceTransformer) -> None:
    with open(input_file, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    textos = [row.get("text", "").strip() for row in comentarios]
    print(f"\nGerando embeddings de {len(textos)} comentários coletados...")
    X = encoder.encode(textos, show_progress_bar=True, batch_size=64)

    print("Classificando comentarios reais...")
    probabilidades = clf.predict_proba(X)
    classes = list(clf.classes_)
    idx_nocivo = classes.index("nocivo")

    total_nocivos = 0

    for row, probs in zip(comentarios, probabilidades):
        prob_nocivo = probs[idx_nocivo]

        if prob_nocivo >= LIMIAR_CONFIANCA:
            categoria_final = "nocivo"
            total_nocivos += 1
        else:
            categoria_final = "nao_nocivo"

        confianca = "alta" if prob_nocivo > 0.70 or prob_nocivo < 0.30 else "media"

        row["categoria"] = categoria_final
        row["confianca"] = confianca
        row["justificativa"] = f"classificador binario embeddings (prob_nocivo={prob_nocivo:.2f})"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comentarios[0].keys())
        writer.writeheader()
        writer.writerows(comentarios)

    pct_nocivos = (total_nocivos / len(comentarios)) * 100
    print(f"\nTransform FASE 1 concluído: {output_file}")
    print(f"Total processado: {len(comentarios)}")
    print(f"Filtrados como NOCIVOS: {total_nocivos} ({pct_nocivos:.1f}%)")
    print(f"Filtrados como NÃO-NOCIVOS: {len(comentarios) - total_nocivos}")


if __name__ == "__main__":
    print(f"Carregando modelo de embeddings: {MODELO_EMBEDDING}...")
    encoder = SentenceTransformer(MODELO_EMBEDDING)

    textos_treino, categorias_treino = carregar_told_br_binario(TOLD_BR_FILE)
    clf = treinar_classificador_binario(textos_treino, categorias_treino, encoder)

    classificar_comentarios(INPUT_FILE, OUTPUT_FILE, clf, encoder)