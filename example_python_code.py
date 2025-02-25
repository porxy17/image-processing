import cv2
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('best.pt')  # Yolun doðru olduðundan emin olun

# Kamera kaynaðýný aç (0 varsayýlan kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açýlamadý!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alýnamadý, çýkýlýyor...")
        break

    # YOLO modelini kullanarak tahmin yap
    results = model(frame)

    # Sonuçlarý görüntü üzerine çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow('YOLO Detection', annotated_frame)

    # 'q' tuþuna basýldýðýnda çýk
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynaklarý serbest býrak
cap.release()
cv2.destroyAllWindows()
