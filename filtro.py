"""
Classifica os comentários coletados e filtra apenas os identificados
como discurso de ódio direcionado a grupos vulneráveis (interseccional:
gênero, raça, sexualidade) — conforme o escopo do projeto VIRAL.

Usa a API gratuita do Google Gemini.
"""

import csv
import json
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit(
        "Erro: variável GEMINI_API_KEY não encontrada.\n"
        "Adicione no .env:\nGEMINI_API_KEY=sua_chave_aqui"
    )

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

INPUT_FILE = "data/scraped/comentarios_roda_viva_erika_hilton_20260711.csv"
OUTPUT_FILE = "data/scraped/comentarios_nocivos_filtrados.csv"

PROMPT_SISTEMA = """Você é um classificador de discurso de ódio em português brasileiro,
especializado em violência política interseccional (gênero, raça, sexualidade).

Classifique o comentário a seguir em UMA das categorias:
- homofobia_transfobia: ataques por orientação sexual ou identidade de gênero
- racismo: ataques por raça/etnia
- misoginia: ataques por gênero (sexismo, misoginia contra mulheres)
- xenofobia: ataques por origem/nacionalidade/regionalismo
- outros_odio: ódio/ofensa que não se encaixa nas categorias acima
- nenhum: comentário sem conteúdo ofensivo

Responda APENAS em JSON válido, sem texto adicional, sem markdown:
{"categoria": "...", "confianca": "alta|media|baixa", "justificativa_breve": "..."}

Comentário: """


def classificar_comentario(texto: str) -> dict:
    try:
        resposta = model.generate_content(PROMPT_SISTEMA + texto)
        conteudo = resposta.text.strip()
        conteudo = conteudo.replace("```json", "").replace("```", "").strip()
        return json.loads(conteudo)
    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"  Erro ao classificar: {e}")
        return {"categoria": "erro", "confianca": "baixa", "justificativa_breve": str(e)}


def processar_csv(input_file: str, output_file: str) -> None:
    with open(input_file, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    print(f"Total de comentários a classificar: {len(comentarios)}")

    nocivos = []
    erros = []
    for i, row in enumerate(comentarios, 1):
        texto = row.get("text", "").strip()
        if not texto:
            continue

        classificacao = classificar_comentario(texto)
        categoria = classificacao.get("categoria", "erro")

        # erro de API é diferente de "sem ódio" — nunca descartar em silêncio
        if categoria == "erro":
            erros.append(row)
        elif categoria != "nenhum":
            row["categoria"] = categoria
            row["confianca"] = classificacao.get("confianca", "")
            row["justificativa"] = classificacao.get("justificativa_breve", "")
            nocivos.append(row)

        if i % 50 == 0:
            print(f"  Processados: {i}/{len(comentarios)} | Nocivos até agora: {len(nocivos)}")

        # tier gratuito do Gemini: ~15 requisições/minuto → 2s de folga entre chamadas
        time.sleep(2.0)

    if erros:
        erros_file = output_file.replace(".csv", "_erros.csv")
        with open(erros_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=erros[0].keys())
            writer.writeheader()
            writer.writerows(erros)
        print(f"ATENÇÃO: {len(erros)} comentários falharam por erro de API. Salvos em: {erros_file}")
        print("  Rode o script novamente usando esse arquivo como input pra reprocessar só esses.")

    if not nocivos:
        print("Nenhum comentário nocivo identificado.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=nocivos[0].keys())
        writer.writeheader()
        writer.writerows(nocivos)

    print(f"\nSalvo em: {output_file} ({len(nocivos)} comentários nocivos de {len(comentarios)} totais)")


if __name__ == "__main__":
    processar_csv(INPUT_FILE, OUTPUT_FILE)