"""
ETL — Etapa L (Load)
Lê o CSV já classificado (etapa Transform) e gera o dataset final,
separando: nocivos (pra análise), erros (pra reprocessar) e um resumo.
"""

import csv
import os

INPUT_FILE = "data/processed/comentarios_classificados.csv"
OUTPUT_NOCIVOS = "data/final/comentarios_nocivos.csv"
OUTPUT_ERROS = "data/final/comentarios_com_erro.csv"

CATEGORIAS_NOCIVAS = {"homofobia_transfobia", "racismo", "misoginia", "xenofobia", "outros_odio"}
CATEGORIAS_ERRO = {"erro_rate_limit", "erro_outro"}


def carregar(input_file: str, output_file: str) -> None:
    with open(input_file, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    nocivos = [r for r in linhas if r.get("categoria") in CATEGORIAS_NOCIVAS]
    erros = [r for r in linhas if r.get("categoria") in CATEGORIAS_ERRO]
    neutros = [r for r in linhas if r.get("categoria") == "nenhum"]

    os.makedirs("data/final", exist_ok=True)

    if nocivos:
        with open(OUTPUT_NOCIVOS, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=nocivos[0].keys())
            writer.writeheader()
            writer.writerows(nocivos)

    if erros:
        with open(OUTPUT_ERROS, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=erros[0].keys())
            writer.writeheader()
            writer.writerows(erros)

    print("Resumo do Load:")
    print(f"  Total de comentários: {len(linhas)}")
    print(f"  Nocivos:  {len(nocivos)} → {OUTPUT_NOCIVOS if nocivos else '(nenhum salvo)'}")
    print(f"  Neutros:  {len(neutros)}")
    print(f"  Erros:    {len(erros)} → {OUTPUT_ERROS if erros else '(nenhum)'}")

    if nocivos:
        print("\n  Distribuição por categoria:")
        contagem = {}
        for r in nocivos:
            cat = r["categoria"]
            contagem[cat] = contagem.get(cat, 0) + 1
        for cat, qtd in sorted(contagem.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {qtd}")


if __name__ == "__main__":
    carregar(INPUT_FILE, OUTPUT_NOCIVOS)