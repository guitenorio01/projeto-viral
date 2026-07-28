"""
Análise exploratória dos comentários nocivos identificados.
Lê data/final/comentarios_nocivos.csv (saída da etapa Load do ETL)
e gera gráficos + resumo estatístico.
"""

import csv
import os
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt

INPUT_FILE = "data/final/comentarios_nocivos.csv"
OUTPUT_DIR = "data/final/graficos"


def carregar_dados(input_file: str) -> list[dict]:
    with open(input_file, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grafico_por_categoria(dados: list[dict], output_dir: str) -> None:
    contagem = Counter(row["categoria"] for row in dados)
    categorias = list(contagem.keys())
    valores = list(contagem.values())

    plt.figure(figsize=(9, 5))
    plt.bar(categorias, valores, color="#c0392b")
    plt.title("Comentários nocivos por categoria — Roda Viva (Erika Hilton)")
    plt.xlabel("Categoria")
    plt.ylabel("Quantidade de comentários")
    plt.xticks(rotation=20)
    plt.tight_layout()

    caminho = os.path.join(output_dir, "nocivos_por_categoria.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Salvo: {caminho}")


def grafico_por_dia(dados: list[dict], output_dir: str) -> None:
    contagem_dia = Counter()
    for row in dados:
        data_str = row.get("published_at", "")
        if not data_str:
            continue
        try:
            data = datetime.strptime(data_str[:10], "%Y-%m-%d")
            contagem_dia[data.strftime("%d/%m")] += 1
        except ValueError:
            continue

    dias_ordenados = sorted(
        contagem_dia.items(),
        key=lambda x: datetime.strptime(x[0], "%d/%m"),
    )
    labels = [d for d, _ in dias_ordenados]
    valores = [v for _, v in dias_ordenados]

    plt.figure(figsize=(11, 5))
    plt.plot(labels, valores, marker="o", color="#8e44ad")
    plt.title("Comentários nocivos ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Quantidade de comentários nocivos")
    plt.xticks(rotation=45)
    plt.tight_layout()

    caminho = os.path.join(output_dir, "nocivos_por_data.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Salvo: {caminho}")


def grafico_confianca(dados: list[dict], output_dir: str) -> None:
    contagem = Counter(row.get("confianca", "não informado") for row in dados)

    plt.figure(figsize=(7, 7))
    plt.pie(
        contagem.values(),
        labels=contagem.keys(),
        autopct="%1.1f%%",
        colors=["#e74c3c", "#f39c12", "#95a5a6"],
    )
    plt.title("Nível de confiança da classificação")
    plt.tight_layout()

    caminho = os.path.join(output_dir, "nivel_confianca.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Salvo: {caminho}")


def resumo_textual(dados: list[dict]) -> None:
    total = len(dados)
    contagem = Counter(row["categoria"] for row in dados)

    print("\n=== RESUMO DA ANÁLISE ===")
    print(f"Total de comentários nocivos: {total}")
    print("\nDistribuição por categoria:")
    for cat, qtd in contagem.most_common():
        pct = (qtd / total) * 100
        print(f"  {cat}: {qtd} ({pct:.1f}%)")


def analisar(input_file: str, output_dir: str) -> None:
    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado: {input_file}")
        print("Rode antes: python etl/transform.py  e  python etl/load.py")
        return

    dados = carregar_dados(input_file)
    if not dados:
        print("Nenhum comentário nocivo encontrado no arquivo.")
        return

    os.makedirs(output_dir, exist_ok=True)

    grafico_por_categoria(dados, output_dir)
    grafico_por_dia(dados, output_dir)
    grafico_confianca(dados, output_dir)
    resumo_textual(dados)


if __name__ == "__main__":
    analisar(INPUT_FILE, OUTPUT_DIR)