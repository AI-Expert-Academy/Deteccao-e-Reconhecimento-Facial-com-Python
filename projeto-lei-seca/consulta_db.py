import cv2
import face_recognition
import faiss
import numpy as np

index = faiss.read_index("./database/cnh_imgs.faiss")

#foto_query = cv2.imread('./database/imgs/test/10462274756.jpg')
#foto_query = cv2.imread('./database/imgs/test/10462274756_300x300.jpg')
#foto_query = cv2.imread('./database/imgs/test/euwebcam.png')
foto_query = cv2.imread('./database/imgs/test/carina.png')
foto_query_rgb = cv2.cvtColor(foto_query, cv2.COLOR_BGR2RGB)
localizacao_face = face_recognition.face_locations(foto_query_rgb, model='cnn')
query_enconding_face = face_recognition.face_encodings(foto_query_rgb, localizacao_face)[0].astype("float32").reshape(1, -1)

k = 5  # número de matches
distances, indices = index.search(query_enconding_face, k)

for rank, idx in enumerate(indices[0]):
    dist = distances[0][rank]
    print(f"Match #{rank+1}: (distância: {np.sqrt(dist):.4f}) - Índice: {idx}")
