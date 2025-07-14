import faiss
import numpy as np
import face_recognition
import cv2

# Supomos que você tenha 1 milhão de embeddings de 512 dimensões
# embeddings = np.random.rand(1000000, 512).astype('float32')
foto_eu = cv2.imread('eu.jpg')
foto_eu_rgb = cv2.cvtColor(foto_eu, cv2.COLOR_BGR2RGB)
localizacao_face = face_recognition.face_locations(foto_eu_rgb, model='cnn')
#print("Localização da face:", localizacao_face)
enconding_face = face_recognition.face_encodings(foto_eu_rgb, localizacao_face)
#enconding_face = enconding_face[0]
#print("Embedding da face:", enconding_face)

embeddings = np.array(enconding_face).astype("float32")

# Cria índice
index = faiss.IndexFlatL2(128)
index.add(embeddings)

print(f"Indexado {index.ntotal} rostos")

# 4. Fazer uma consulta (nova imagem)
query_image = face_recognition.load_image_file("carina.png")
#query_image = face_recognition.load_image_file("euwebcam.png")
localizacao_face_query_image = face_recognition.face_locations(query_image, model='cnn')
#print("Localização da face na imagem de consulta:", localizacao_face_query_image)
query_encoding = face_recognition.face_encodings(query_image, localizacao_face_query_image)[0].astype("float32").reshape(1, -1)
#print("Embedding da face na imagem de consulta:", query_encoding)

# 5. Buscar os rostos mais parecidos
k = 1  # número de matches
distances, indices = index.search(query_encoding, k)
#print(index.search(query_encoding, k))

#print(indices)
#print(indices[0])

# 6. Exibir os resultados
for rank, idx in enumerate(indices[0]):
    dist = distances[0][rank]
    print(f"Match #{rank+1}: (distância: {np.sqrt(dist):.4f})")
    #print(type(idx))
    vetor_retornado = index.reconstruct(int(idx))
    #print(f"Vetor retornado: {vetor_retornado}")
    distancia_face_recognition = face_recognition.face_distance(vetor_retornado, query_encoding)[0]
    print(f"Distância retornada pelo face_recognition: {distancia_face_recognition:.4f}")
