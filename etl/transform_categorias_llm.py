"""
ETL — Etapa T (Transform), Fase 2: categorização fina via LLM
Lê a saída da Fase 1 (classificação binária local) e usa o Gemini
para decidir a categoria específica (homofobia, racismo, misoginia,
xenofobia, outros) SÓ nos comentários já marcados como nocivos.
Isso reduz drasticamente o volume de chamadas de API, tornando viável
usar o tier gratuito mesmo com cota baixa.
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
model = genai.GenerativeModel("gemini-2.5-flash-lite")

INPUT_FILE = "data/processed/comentarios_binario.csv"
OUTPUT_FILE = "data/processed/comentarios_classificados.csv"

PROMPT_SISTEMA = """Você é um classificador de discurso de ódio em português brasileiro,
especializado em violência política interseccional (gênero, raça, sexualidade).
Este comentário já foi identificado como potencialmente nocivo. Sua tarefa é
apenas decidir a categoria mais específica.

Categorias:
- homofobia_transfobia: ataques por orientação sexual ou identidade de gênero
- racismo: ataques por raça/etnia
- misoginia: ataques por gênero (sexismo, misoginia contra mulheres)
- xenofobia: ataques por origem/nacionalidade/regionalismo
- outros_odio: nocivo, mas não se encaixa nas categorias acima
- nenhum: na verdade não é nocivo (falso positivo da triagem anterior)

Responda APENAS em JSON válido, sem texto adicional, sem markdown:
{"categoria": "...", "confianca": "alta|media|baixa"}

Comentário: """

MAX_TENTATIVAS = 3


def classificar_categoria(texto: str) -> dict:
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = model.generate_content(PROMPT_SISTEMA + texto)
            conteudo = resposta.text.strip().replace("```json", "").replace("```", "").strip()
            resultado = json.loads(conteudo)
            resultado["status"] = "ok"
            return resultado
        except Exception as e:
            erro_str = str(e)
            eh_rate_limit = "429" in erro_str or "ResourceExhausted" in erro_str or "quota" in erro_str.lower()

            if eh_rate_limit and tentativa < MAX_TENTATIVAS:
                espera = 20 * tentativa
                if tentativa == 1:
                    print(f"    ERRO COMPLETO (diagnóstico): {erro_str[:500]}")
                print(f"    Rate limit — tentativa {tentativa}/{MAX_TENTATIVAS}, aguardando {espera}s...")
                time.sleep(espera)
                continue

            return {
                "categoria": "erro_rate_limit" if eh_rate_limit else "erro_outro",
                "confianca": "",
                "status": "falhou",
            }

    return {"categoria": "erro_outro", "confianca": "", "status": "falhou"}


def categorizar(input_file: str, output_file: str) -> None:
    with open(input_file, encoding="utf-8") as f:
        comentarios = list(csv.DictReader(f))

    nocivos = [r for r in comentarios if r.get("nocivo_binario") == "nocivo"]
    neutros = [r for r in comentarios if r.get("nocivo_binario") != "nocivo"]

    print(f"Total de comentários: {len(comentarios)}")
    print(f"Marcados como nocivos pela Fase 1 (serão categorizados via LLM): {len(nocivos)}")
    print(f"Já neutros (não precisam de chamada de API): {len(neutros)}")

    for row in neutros:
        row["categoria"] = "nenhum"
        row["confianca"] = row.get("confianca_binaria", "")
        row["justificativa"] = "classificador binário local"

    falhas = 0
    for i, row in enumerate(nocivos, 1):
        texto = row.get("text", "").strip()
        resultado = classificar_categoria(texto)

        if resultado["status"] == "falhou":
            falhas += 1

        row["categoria"] = resultado.get("categoria", "erro_outro")
        row["confianca"] = resultado.get("confianca", "")
        row["justificativa"] = "classificador embeddings (fase 1) + LLM (fase 2)"

        if i % 10 == 0:
            print(f"  Categorizados: {i}/{len(nocivos)} | Falhas até agora: {falhas}")

        time.sleep(5.0)  # tier gratuito: ~15 RPM, indo com folga

    todos = neutros + nocivos
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=todos[0].keys())
        writer.writeheader()
        writer.writerows(todos)

    print(f"\nFase 2 concluída: {output_file}")
    print(f"Total: {len(todos)} | Falhas de API: {falhas}")
    if falhas > 0:
        print("Comentários com categoria 'erro_*' podem ser reprocessados rodando este script de novo.")


if __name__ == "__main__":
    categorizar(INPUT_FILE, OUTPUT_FILE)