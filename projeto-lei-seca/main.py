import re
import cv2 # OpenCV
import easyocr
import face_recognition
import faiss
import imutils
import numpy as np
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

def normalizar_ocr(texto_ocr: str) -> str:
  substituicoes = {
      '@': 'Q',
      '0': 'O',
      '1': 'I',
      '8': 'B',
      '$': 'S',
  }

  texto_corrigido = texto_ocr.upper()
  texto_corrigido_chararray = list(texto_corrigido)
  for i, char in enumerate(texto_corrigido_chararray):
    if i == 0 or i == 1 or i == 2 or i == 4:
      if char in substituicoes:
        texto_corrigido_chararray[i] = substituicoes[char]

  return "".join(texto_corrigido_chararray)

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

uploaded_file = st.file_uploader("Faça upload de uma imagem com uma placa", type=["jpg", "jpeg", "png"])

if uploaded_file:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # 2 Containers lado a lado: Um para detecção de placa e outro para reconhecimento
  colDeteccaoPlaca, colReconhecimentoPlaca = st.columns(2)
  
  with colDeteccaoPlaca:
    reader = easyocr.Reader(['pt'])
    results = reader.readtext(image)
    
    st.write("Detecção de placa:")
    pattern = r'^[A-Z0-9]{3}[0-9][A-Z0-9][0-9]{2}$'
    algumaPlacaDetectada = False
    for bbox, text, conf in results:
      if conf > 0.5:
        placa = text.upper().strip()
        placa = normalizar_ocr(placa)

        print(f"Texto detectado: {placa} (confiança: {conf:.2f})")

        if re.fullmatch(pattern, placa):
          algumaPlacaDetectada = True
          (tl, tr, br, bl) = bbox
          tl = tuple(map(int, tl))
          br = tuple(map(int, br))
          cv2.rectangle(image, tl, br, (0, 255, 0), 2)
          st.image(image, caption="Imagem carregada", use_container_width=True)
          print(f"✅ {placa} é válida")
          print(f"Texto detectado: {placa} (confiança: {conf:.2f})")

          with colReconhecimentoPlaca:
            st.write("Reconhecimento de placa:")
            st.text_input("Placa detectada:", value=placa, key="placa_detectada")
            st.button("Conferir via placa", key="conferir_placa")
            st.badge("IPVA em dia", color="gray", icon="✅")
            st.badge("Multas em aberto", color="gray", icon="🚨")
            st.badge("Carro clonado", color="gray", icon="🚨")
        else:
          print(f"❌ {placa} é inválida")
    
    if not algumaPlacaDetectada:
      st.write("Nenhuma placa detectada.")
      st.button("Conferir via placa", key="conferir_placa")