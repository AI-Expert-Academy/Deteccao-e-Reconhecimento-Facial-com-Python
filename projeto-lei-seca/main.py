import cv2 # OpenCV
import numpy as np
import tempfile
import os
import streamlit as st

from PIL import Image

def detectaFacesSSD(net, imagem, tamanho = 300):
  (h, w) = imagem.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (tamanho,tamanho)), 1.0, (tamanho,tamanho), (104.0, 177.0, 123.0))
  net.setInput(blob)
  deteccoes = net.forward()
  conf_min = 0.5

  for i in range(0, deteccoes.shape[2]):
    confianca = deteccoes[0,0,i,2]
    
    if confianca > conf_min:
      box = deteccoes[0,0,i,3:7] * np.array([w,h,w,h])
      (startX, startY, endX, endY) = box.astype("int")
      
      text_conf = "{:.2f}%".format(confianca * 100)
      print(text_conf)
      cv2.rectangle(imagem, (startX, startY), (endX, endY), (0,255,0), 2)
      cv2.putText(imagem, text_conf, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

      return imagem
    else:
      print("Nenhuma face detectada com confiança suficiente.")
      return None


arquivo_modelo = './res10_300x300_ssd_iter_140000.caffemodel'
arquivo_prototxt = './deploy.prototxt.txt'
network = cv2.dnn.readNetFromCaffe(arquivo_prototxt, arquivo_modelo)

st.title("Lei Seca do Futuro - Detecção de Faces e placas")

# Interface da câmera
img_file_buffer = st.camera_input("Tire uma foto")

if img_file_buffer is not None:
    # Converte para imagem OpenCV
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Mostra a imagem capturada
    st.image(img_array, caption="Imagem capturada", use_container_width=True)

    # Mostra a face detectada
    st.write("Detecção de face:")
    faceDetected = detectaFacesSSD(network, img_array)
    if faceDetected is not None:
      faceDetectedArray = np.array(faceDetected)
      st.image(faceDetectedArray, caption="Rosto detectado", use_container_width=True)
    else:
      st.write("Nenhuma face detectada.")
    