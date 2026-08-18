"""
ETL — Fase 2 (Categorização de Nocivos via Gemini API - Cota Free Ajustada)
Ajustado para respeitar a cota gratuita de 5 requisições por minuto (5 RPM).
"""

import csv
import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

INPUT_FILE = "data/processed/comentarios_classificados.csv"
OUTPUT_FILE = "data/processed/comentarios_classificados.csv"

MODELO_GEMINI = "gemini-2.0-flash-exp"
BATCH_SIZE = 15  # 25 itens por lote para reduzir chamadas totais
PAUSA_ENTRE_LOTES = 13  # 13s entre lotes garante ~4.6 requisições/minuto (dentro do limite de 5)

PROMPT_SISTEMA = """
Você é um classificador especializado em análise de discurso de ódio em comentários de redes sociais (YouTube).
Sua tarefa é analisar uma lista de comentários pré-filtrados como nocivos e classificar CADA UM em exatamente UMA das categorias abaixo:

CATEGORIAS PERMITIDAS:
1. misoginia: Ataques, ofensas, preconceito ou machismo contra mulheres.
2. racismo: Ofensas, estereótipos ou discriminação baseada em raça/cor.
3. homofobia_transfobia: Ataques à comunidade LGBTQIA+, orientação sexual ou identidade de gênero.
4. xenofobia: Preconceito/ataques contra pessoas de determinada região, estado ou país (ex: ataques a nordestinos, estrangeiros).
5. outros_odio: Insultos graves, obscenidades, ameaças ou agressividade explícita que não se enquadram claramente nas categorias anteriores.

Sua resposta DEVE ser estritamente um JSON no seguinte formato (uma lista de objetos):
[
  {
    "id": 0,
    "categoria": "misoginia",
    "justificativa": "Ataque direto de cunho machista à entrevistada."
  }
]
"""


def processar_lote_com_retry(client, batch, max_tentativas=3):
    """Envia um lote para o Gemini com mecanismo de retry se bater na cota."""
    comentarios_texto = "\n".join([f"ID {item['id_local']}: \"{item['text']}\"" for item in batch])
    prompt_usuario = f"Classifique os seguintes comentários:\n\n{comentarios_texto}"

    for tentativa in range(1, max_tentativas + 1):
        try:
            response = client.models.generate_content(
                model=MODELO_GEMINI,
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=PROMPT_SISTEMA,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            resultados = json.loads(response.text)
            return {item["id"]: item for item in resultados}

        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                espera = 15 * tentativa
                print(f"   ⏳ Cota atingida (429). Aguardando {espera}s para tentar novamente (Tentativa {tentativa}/{max_tentativas})...")
                time.sleep(espera)
            else:
                print(f"   ⚠️ Erro inesperado no lote: {e}")
                break

    return {}


def executar_fase2():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("A variável GEMINI_API_KEY não foi encontrada no arquivo .env!")

    client = genai.Client(api_key=api_key)

    with open(INPUT_FILE, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    nocivos_indices = [
        idx for idx, c in enumerate(comentarios) if c.get("categoria") == "nocivo"
    ]

    print(f"Total de comentários na base: {len(comentarios)}")
    print(f"Comentários para categorizar via Gemini: {len(nocivos_indices)}")

    itens_para_processar = [
        {"id_local": idx, "text": comentarios[idx].get("text", "")}
        for idx in nocivos_indices
    ]

    total_lotes = (len(itens_para_processar) + BATCH_SIZE - 1) // BATCH_SIZE
    tempo_estimado_min = (total_lotes * PAUSA_ENTRE_LOTES) / 60

    print(f"\nIniciando categorização refinada ({total_lotes} lotes de {BATCH_SIZE} itens)...")
    print(f"⏱️ Tempo estimado total: ~{tempo_estimado_min:.1f} minutos para respeitar a cota gratuita.\n")

    for i in range(0, len(itens_para_processar), BATCH_SIZE):
        batch = itens_para_processar[i : i + BATCH_SIZE]
        num_lote = (i // BATCH_SIZE) + 1
        print(f"Processando lote {num_lote}/{total_lotes} ({len(batch)} itens)...")

        resultados_lote = processar_lote_com_retry(client, batch)

        for item in batch:
            idx = item["id_local"]
            res = resultados_lote.get(idx) or resultados_lote.get(str(idx))

            if res and "categoria" in res:
                comentarios[idx]["categoria"] = res["categoria"]
                comentarios[idx]["justificativa"] = f"Gemini LLM: {res.get('justificativa', '')}"
                comentarios[idx]["confianca"] = "alta"
            else:
                comentarios[idx]["categoria"] = "outros_odio"
                comentarios[idx]["justificativa"] = "Gemini LLM (fallback por timeout)"

        time.sleep(PAUSA_ENTRE_LOTES)

    for c in comentarios:
        if c["categoria"] == "nao_nocivo":
            c["categoria"] = "nenhum"

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comentarios[0].keys())
        writer.writeheader()
        writer.writerows(comentarios)

    print(f"\n✅ FASE 2 concluída com sucesso! CSV atualizado em: {OUTPUT_FILE}")


if __name__ == "__main__":
    executar_fase2()