import os
import cv2
import numpy as np

# === 1) Paths ===
PB_PATH     = "frozen_model.pb"
LABELS_PATH = "modelo_teachable/labels.txt"

for p, desc in [(PB_PATH, "frozen_model.pb"), (LABELS_PATH, "labels.txt")]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No encontré {desc} en: {p}")

# === 2) Carga del modelo ===
net = cv2.dnn.readNetFromTensorflow(PB_PATH)

# === 3) Lectura de etiquetas (quitando índices numéricos) ===
labels = []
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        name = parts[1] if len(parts) > 1 else parts[0]
        labels.append(name.upper())

# Ahora labels debe ser:
# ["FABRICIO", "TOMATODO", "LAPICERO", "CELULAR", "DIEGO", "SEBASTIAN", "ETSON"]

# === 4) Colores por clase ===
colors = {
    "FABRICIO":  (0, 255, 0),
    "TOMATODO":  (255, 0, 255),
    "LAPICERO":  (0, 255, 255),
    "CELULAR":   (255, 0, 0),
    "DIEGO":     (128, 0, 128),
    "SEBASTIAN": (0, 128, 255),
    "ETSON":     (0, 200, 200),
}
default_color = (200, 200, 200)

# === 5) Iniciar cámara ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No pude abrir la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 6) Preprocesado ===
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1/255.0,
        size=(224, 224),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)

    # === 7) Inferencia ===
    preds = net.forward()[0]   # vector de probabilidades

    # === 8) Selección de clase ===
    idx   = int(np.argmax(preds))
    label = labels[idx]
    prob  = float(preds[idx])

    # === 9) Dibujo con color correcto ===
    color = colors.get(label, default_color)
    texto = f"{label} ({prob*100:.1f}%)"
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    cv2.imshow("Detección 7 Clases", frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
