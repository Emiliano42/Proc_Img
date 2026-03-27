print("Hola mundo")

import cv2

img = cv2.imread("entre-ciel-et-terre.jpg")
Img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Img_muestreada = Img_gris

#--Reducir muestreo a la mitad

# Copia los pixeles de las posiciones PARES y los pone en las posiciones IMPARES
Img_muestreada[::2, ::2] = Img_gris[::2, ::2]  # Nose si esta linea es al pedo o no
Img_muestreada[1::2, ::2] = Img_gris[::2, ::2]     # copia vertical
Img_muestreada[::2, 1::2] = Img_gris[::2, ::2]     # copia horizontal
Img_muestreada[1::2, 1::2] = Img_gris[::2, ::2]    # copia diagonal

#--Cuantificar la imagen a la mitad
img_cuantificada = (Img_gris // 2) *2  #Si solo divido me quedo con la mitad de los niveles(o el numero por el que divida), osea me quedo solo con la parte mas oscura, por lo que es necesario volver a multiplicar asi vuelvo a acceder a la mitad mas clara(0 = negro, 255 = Blanco), consiguiendo asi AGRUPAR valores/niveles

#Normal----
#cv2.imshow("aura", img)
#print("aura", img.shape)
#print("aura:",img[0,0]) #Me da los distintos tonos del pixel[0,0], 0 = negro, 255 = Blanco, Intermedio = escala de gris

#Gris----
cv2.imshow("FotoGris", Img_gris)
print("imagen gris:", Img_gris.shape)
print("imagen gris:", Img_gris[0,0])
#print(Img_gris.dtype)

#Muestreada----
cv2.imshow("Muestreada", Img_muestreada)
print("imagen muestreada:", Img_muestreada.shape)

#Cuantificada----
cv2.imshow("Cuantificada", img_cuantificada)
print("imagen cuantificada:", img_cuantificada.shape)
print("imagen cuantificada:", img_cuantificada[0,0])
cv2.waitKey(0)
cv2.destroyAllWindows()