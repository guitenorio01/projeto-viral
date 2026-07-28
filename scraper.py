"""
Scraper de comentários do YouTube — caso de validação VIRAL
Vídeo: Erika Hilton no Programa Roda Viva (30/03/2026)
https://www.youtube.com/watch?v=61_plHadgJU
"""

import csv
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Configuração ──────────────────────────────────────────────
load_dotenv()  # lê o arquivo .env na raiz do projeto

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    sys.exit(
        "Erro: variável YOUTUBE_API_KEY não encontrada.\n"
        "Crie um arquivo .env na raiz do projeto com:\n"
        "YOUTUBE_API_KEY=sua_chave_aqui"
    )

VIDEO_ID = "61_plHadgJU"
OUTPUT_DIR = "data/scraped"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"comentarios_roda_viva_erika_hilton_{datetime.now().strftime('%Y%m%d')}.csv",
)
MAX_RESULTS_PER_PAGE = 100  # máximo permitido pela API

def buscar_replies_completas(youtube, parent_id: str) -> list[dict]:
    """Busca TODAS as respostas de um comentário via comments().list,
    já que commentThreads().list só traz uma amostra parcial."""
    replies = []
    next_page_token = None

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            r_snippet = item["snippet"]
            replies.append({
                "comment_id": item["id"],
                "author": r_snippet.get("authorDisplayName", ""),
                "text": r_snippet.get("textDisplay", ""),
                "like_count": r_snippet.get("likeCount", 0),
                "published_at": r_snippet.get("publishedAt", ""),
                "is_reply": True,
                "parent_id": parent_id,
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return replies


# ── Coleta ────────────────────────────────────────────────────
def coletar_comentarios(api_key: str, video_id: str) -> list[dict]:
    youtube = build("youtube", "v3", developerKey=api_key)
    comentarios = []
    next_page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=MAX_RESULTS_PER_PAGE,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",  # ou "time" pra ordem cronológica
            )
            response = request.execute()

            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_id = item["snippet"]["topLevelComment"]["id"]
                comentarios.append({
                    "comment_id": comment_id,
                    "author": top_comment.get("authorDisplayName", ""),
                    "text": top_comment.get("textDisplay", ""),
                    "like_count": top_comment.get("likeCount", 0),
                    "published_at": top_comment.get("publishedAt", ""),
                    "is_reply": False,
                    "parent_id": None,
                })

                # respostas ao comentário — busca completa quando necessário
                total_replies = item["snippet"].get("totalReplyCount", 0)
                replies_embutidas = len(item.get("replies", {}).get("comments", []))

                if total_replies > 0:
                    if total_replies > replies_embutidas:
                        # amostra embutida é parcial → busca todas via comments().list
                        try:
                            replies_completas = buscar_replies_completas(youtube, comment_id)
                            comentarios.extend(replies_completas)
                        except HttpError as e:
                            print(f"  Aviso: falha ao buscar replies completas de {comment_id}: {e}")
                            print("  Usando as replies parciais já disponíveis e seguindo a coleta.")
                            for reply in item.get("replies", {}).get("comments", []):
                                r_snippet = reply["snippet"]
                                comentarios.append({
                                    "comment_id": reply["id"],
                                    "author": r_snippet.get("authorDisplayName", ""),
                                    "text": r_snippet.get("textDisplay", ""),
                                    "like_count": r_snippet.get("likeCount", 0),
                                    "published_at": r_snippet.get("publishedAt", ""),
                                    "is_reply": True,
                                    "parent_id": comment_id,
                                })
                    else:
                        for reply in item["replies"]["comments"]:
                            r_snippet = reply["snippet"]
                            comentarios.append({
                                "comment_id": reply["id"],
                                "author": r_snippet.get("authorDisplayName", ""),
                                "text": r_snippet.get("textDisplay", ""),
                                "like_count": r_snippet.get("likeCount", 0),
                                "published_at": r_snippet.get("publishedAt", ""),
                                "is_reply": True,
                                "parent_id": comment_id,
                            })

            next_page_token = response.get("nextPageToken")
            print(f"Coletados até agora: {len(comentarios)} (página processada com {len(response.get('items', []))} threads)")

            if not next_page_token:
                break

            time.sleep(0.5)  # evita bater rate limit

        except HttpError as e:
            if "commentsDisabled" in str(e):
                print("Comentários desabilitados neste vídeo.")
                break
            print(f"ERRO FATAL na paginação principal (parou em {len(comentarios)} comentários): {e}")
            break

    return comentarios


def salvar_csv(comentarios: list[dict], output_file: str) -> None:
    if not comentarios:
        print("Nenhum comentário coletado.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comentarios[0].keys())
        writer.writeheader()
        writer.writerows(comentarios)

    print(f"Salvo em: {output_file} ({len(comentarios)} comentários)")


if __name__ == "__main__":
    comentarios = coletar_comentarios(API_KEY, VIDEO_ID)
    salvar_csv(comentarios, OUTPUT_FILE)