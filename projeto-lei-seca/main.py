import cv2 # OpenCV
import face_recognition
import faiss
import imutils
import numpy as np
import tempfile
import os
import streamlit as st
import logging
import pytesseract

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
  handler = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)

logger.info("Início do carregamento de base de dados de imagens de CNH.")
index = faiss.read_index("./database/cnh_imgs.faiss")
imgs_cnh = os.listdir('./database/imgs/')
logger.info("Fim do carregamento de base de dados de imagens de CNH.")

def detectaFacesSSD(net, foto):
  tamanho = 300
  (h, w) = foto.shape[:2]
  blob = cv2.dnn.blobFromImage(foto, 1.0, (tamanho,tamanho), (104.0, 177.0, 123.0))
  net.setInput(blob)
  deteccoes = net.forward()
  conf_min = 0.5

  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0,0,i,2]
    text_conf = "{:.2f}%".format(confianca * 100)
    logger.info(f"Rosto detectado com {text_conf} de confiança.")
    
    if confianca > conf_min:
      box = deteccoes[0,0,i,3:7] * np.array([w,h,w,h])
      (startX, startY, endX, endY) = box.astype("int")
      (startX, startY, endX, endY) = [int(v) for v in (startX, startY, endX, endY)]

      cv2.rectangle(foto, (startX, startY), (endX, endY), (0,255,0), 2)
      # cv2.putText(foto, text_conf, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

      return foto, text_conf
    else:
      logger.info("Nenhuma face detectada com confiança suficiente.")
      return None, "0.00%"
    
def reconheceFace(face_300x300):
  face_query_rgb = cv2.cvtColor(face_300x300, cv2.COLOR_BGR2RGB)
  localizacao_face = face_recognition.face_locations(face_query_rgb, model='cnn')
  
  if len(localizacao_face) != 0:
    query_enconding_face = face_recognition.face_encodings(face_query_rgb, localizacao_face)[0].astype("float32").reshape(1, -1)
    k = 3  # número de matches
    distances, indices = index.search(query_enconding_face, k)
    return distances, indices
  else:
    logger.info("Nenhuma face detectada para reconhecimento.")
    logger.debug("Nenhuma face detectada para reconhecimento utilizando modelo CNN da biblioteca face_recognition.")
    return None, None

logger.info("Início do carregamento do modelo de detecção de faces.")
arquivo_modelo = './res10_300x300_ssd_iter_140000.caffemodel'
arquivo_prototxt = './deploy.prototxt.txt'
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)
logger.info("Fim do carregamento do modelo de detecção de faces.")

st.title("Lei Seca do Futuro - Detecção de Faces e Placas")

# Interface da câmera
imgFileBufferFace = st.camera_input("Tire uma foto", key="camera_face")

if imgFileBufferFace is not None:
  # Converte para imagem OpenCV
  img = Image.open(imgFileBufferFace)
  img_array = np.array(img)
  
  img_array_300x300 = imutils.resize(img_array, width=300, height=300)

  # Mostra a imagem capturada
  st.image(img_array, caption="Imagem capturada", use_container_width=True)

  # 2 Containers lado a lado: Um para detecção de face e outro para reconhecimento facial
  colDeteccaoFace, colReconhecimentoFace = st.columns(2)

  faceDetectadaComBoundingBox = None

  # Mostra a face detectada
  with colDeteccaoFace:
    st.write("Detecção de face:")
    faceDetectadaComBoundingBox, textoConfianca = detectaFacesSSD(network, img_array_300x300)
    faceDetectadaArray = np.array(faceDetectadaComBoundingBox)
    if faceDetectadaComBoundingBox is not None:
      caption = f"Rosto detectado com {textoConfianca} de confiança"
      st.image(faceDetectadaArray, caption=caption)
      with colReconhecimentoFace:
        st.write("Reconhecimento facial:")
        distancias, indices = reconheceFace(img_array_300x300)
        
        if distancias is not None or indices is not None:
          # Debug output
          for rank, idx in enumerate(indices[0]):
            dist = distancias[0][rank]
            logger.info(f"Match #{rank+1}: (distância: {np.sqrt(dist):.4f}) - Índice: {idx}")
          
          distanciasNormalizadas = [np.sqrt(dist) for dist in distancias[0]]
          index_of_min = np.argmin(distanciasNormalizadas)
          if distanciasNormalizadas[index_of_min] > 0.55:  # Limite de distância para considerar um match
            st.write("Nenhum rosto reconhecido com confiança suficiente.")
            st.button("Conferir via CPF", key="conferir_cpf")
          else:
            colFoto, colInfo = st.columns(2)
            with colFoto:
              caption = f"Match com probabilidade(distância) de {distanciasNormalizadas[index_of_min]*100:.2f}% em potencial com base dados"
              st.image(f"./database/imgs/{imgs_cnh[index_of_min]}", caption=caption)
            with colInfo:
              st.badge("CNH Ok", color="gray", icon="✅")
              st.badge("Multas em aberto", color="gray", icon="🚨")
              st.badge("Mandado judicial em aberto", color="gray", icon="🚨")
              st.button("Mais detalhes", key="mais_detalhes")
              st.button("Conferir via CPF", key="conferir_cpf")
        else:
          st.write("Nenhum rosto reconhecido.")
          st.button("Conferir via CPF", key="conferir_cpf")
    else:
      st.write("Nenhuma face detectada.")

imgFileBufferPlaca = st.camera_input("Tire uma foto", key="camera_placa")

if imgFileBufferPlaca is not None:
  # Converte para imagem OpenCV
  img = Image.open(imgFileBufferPlaca)
  img_array = np.array(img)

  img_array_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
  blur = cv2.bilateralFilter(img_array_gray, 11, 17, 17)

  edged = cv2.Canny(blur, 30, 200)

  conts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  conts = imutils.grab_contours(conts) 
  conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8] 

  location = None
  for c in conts:
    peri = cv2.arcLength(c, True)
    aprox = cv2.approxPolyDP(c, 0.02 * peri, True)
    if cv2.isContourConvex(aprox):
      if len(aprox) == 4:
        location = aprox
        break
  
  mask = np.zeros(img_array_gray.shape, np.uint8) 

  img_plate = cv2.drawContours(mask, [location], 0, 255, -1)
  img_plate = cv2.bitwise_and(img_array_gray, img_array_gray, mask=mask)

  (y, x) = np.where(mask==255)
  (beginX, beginY) = (np.min(x), np.min(y))
  (endX, endY) = (np.max(x), np.max(y))

  plate = img_array_gray[beginY:endY, beginX:endX]

  cv2.imwrite("./placa.jpg", plate)

  config_tesseract = "--tessdata-dir tessdata --psm 6"
  text = pytesseract.image_to_string(plate, lang="por", config=config_tesseract)
  print(text)
  text = "".join(character for character in text if character.isalnum())
  print(text)

  # 2 Containers lado a lado: Um para detecção de placa e outro para reconhecimento
  colDeteccaoPlaca, colReconhecimentoPlaca = st.columns(2)

  with colDeteccaoPlaca:
    st.write("Detecção de placa:")

    imgPlacaDetectada = cv2.rectangle(img_array, (beginX, beginY), (endX, endY), (0, 255, 0), 3)

    st.image(imgPlacaDetectada, caption="Imagem capturada", use_container_width=True)

    with colReconhecimentoPlaca:
      st.write("Reconhecimento de placa:")
      st.text_input("Placa detectada:", value=text, key="placa_detectada")
      st.button("Conferir via placa", key="conferir_placa")
      st.badge("IPVA em dia", color="gray", icon="✅")
      st.badge("Multas em aberto", color="gray", icon="🚨")
      st.badge("Carro clonado", color="gray", icon="🚨")