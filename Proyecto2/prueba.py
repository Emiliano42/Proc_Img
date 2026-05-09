import cv2
import os

cap = cv2.VideoCapture("espaguetiGris.mp4")

# Crear carpeta dataset
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

ret, frame_base = cap.read()
frame_base = cv2.GaussianBlur(frame_base, (5, 5), 0)

prev_frame = frame_base.copy()

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # SOLO diferencia acumulada
    diff = cv2.absdiff(frame_base, frame_blur)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 🔥 umbral automático (bien que lo cambiaste)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 🔥 limpieza fuerte
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, None, iterations=3)

    # 🔥 quedarse con el objeto más grande (clave)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # Buscar mejor contorno
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filtrar áreas razonables
        if 500 < area < 7000000: #Filtra bien "piezaNegra": 500 < area < 7000000,  x>100, y>100 (creo, sino era con 150). Muy bien para "espaguetiBlanco": 5000 < area < 7000000 y x>200, y>200.

            x, y, w, h = cv2.boundingRect(cnt)

            # Ignorar zonas problemáticas (ejemplo)
            if x > 100 and y > 100:

                # Quedarse con el mejor contorno válido
                if area > best_area:
                    best_area = area
                    best_cnt = cnt

    # Si encontró alguno válido
    if best_cnt is not None:

        x, y, w, h = cv2.boundingRect(best_cnt)

        # Guardar imagen
        img_name = f"dataset/images/frame_{frame_id}.jpg"
        #cv2.imwrite(img_name, frame)

        # Label YOLO
        h_img, w_img = frame.shape[:2]

        x_center = (x + w/2) / w_img
        y_center = (y + h/2) / h_img
        w_norm = w / w_img
        h_norm = h / h_img

        label_name = f"dataset/labels/frame_{frame_id}.txt"

        #with open(label_name, "w") as f:
         #   f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}")

        # Visual
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        frame_id += 1

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    prev_frame = frame_blur

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()