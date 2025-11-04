import nltk
import time
import random
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configurações ---
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS_PT = set(stopwords.words('portuguese'))

API_KEY = "AIzaSyA_iQnglhGczNLHPcOvE8P_qzFOTrfp0DY"
YOUTUBE = build("youtube", "v3", developerKey=API_KEY)

# --- Funções ---
def limpar_texto(texto):
    """Remove stopwords e converte para minúsculas."""
    return " ".join(p for p in texto.lower().split() if p not in STOPWORDS_PT)

def baixar_comentarios(video_id):
    """Baixa todos os comentários e respostas de um vídeo."""
    comentarios = []
    next_page_token = None

    print("Baixando comentários...")

    while True:
        try:
            req = YOUTUBE.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            res = req.execute()

            for item in res["items"]:
                # Comentário principal
                comentario = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comentarios.append(comentario)

                # Respostas (primeiro nível)
                for reply in item.get("replies", {}).get("comments", []):
                    comentarios.append(reply["snippet"]["textDisplay"])

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break

            # Intervalo para evitar limite de taxa
            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"Erro: {e}. Retentando em 5s...")
            time.sleep(5)

    print(f"{len(comentarios)} comentários baixados.\n")
    return comentarios

def buscar_relevantes(comentarios, query):
    """Calcula a similaridade e retorna comentários mais relevantes."""
    comentarios_limpos = [limpar_texto(c) for c in comentarios]
    query_limpa = [limpar_texto(query)]

    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(comentarios_limpos)
    tfidf_query = vectorizer.transform(query_limpa)

    similaridades = cosine_similarity(tfidf_query, tfidf_docs)[0]

    resultados = [
        (c, s) for c, s in zip(comentarios, similaridades) if s > 0
    ]
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados

# --- Execução principal ---
if __name__ == "__main__":
    link = input("Qual link do seu vídeo? ").strip()
    video_id = link.replace("https://www.youtube.com/watch?v=", "").split("&")[0]
    query = input("Qual sua query? ").strip()

    comentarios = baixar_comentarios(video_id)
    relevantes = buscar_relevantes(comentarios, query)

    if relevantes:
        print("\nComentários mais relevantes:")
        for i, (comentario, score) in enumerate(relevantes[:5], 1):
            print(f"{i}. [{score:.3f}] {comentario}")
    else:
        print("\nNenhum comentário relevante encontrado.")
