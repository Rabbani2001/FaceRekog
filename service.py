from deepface import DeepFace
import numpy as np
import threading

inference_lock = threading.Lock()


def generate_embedding(image_path):
    with inference_lock:

        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

    return result[0]["embedding"]


def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    # a = np.array(a)
    # b = np.array(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)

    # return np.dot(a, b) / (
    #     np.linalg.norm(a) *
    #     np.linalg.norm(b)
    # )


def compare_embeddings(
    current_embedding,
    saved_embeddings
):

    best_username = None
    best_score = -1

    for face in saved_embeddings:

        score = cosine_similarity(
            current_embedding,
            face["embedding"]
        )

        if score > best_score:
            best_score = score
            best_username = face["username"]

    return best_username, best_score