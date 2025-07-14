import os
import cv2
import face_recognition
import numpy as np
import imutils
import faiss

index = faiss.IndexFlatL2(128)

imgs = os.listdir('./database/imgs/')
for img_name in imgs:
    if img_name.endswith('.jpg'):
        img_path = os.path.abspath('./database/imgs/'+img_name)
        img = cv2.imread(img_path)
        img_nparray = np.array(img)
        #print(img_name)
        #print(img_nparray.shape)
        img_nparray = imutils.resize(img_nparray, width=300, height=300)
        cv2.imwrite('./database/imgs/'+img_name, img_nparray)

        img_300x300 = cv2.imread('./database/imgs/'+img_name)
        img_300x300_rgb = cv2.cvtColor(img_300x300, cv2.COLOR_BGR2RGB)
        localizacao_face = face_recognition.face_locations(img_300x300_rgb, model='cnn')
        print("Localização da face:", localizacao_face)
        enconding_face = face_recognition.face_encodings(img_300x300_rgb, localizacao_face)
        print("Enconding da face:", enconding_face)
        embeddings = np.array(enconding_face).astype("float32")
        index.add(embeddings)

faiss.write_index(index, "./database/cnh_imgs.faiss")