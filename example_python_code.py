import cv2
from ultralytics import YOLO

# YOLO modelini y�kle
model = YOLO('best.pt')  # Yolun do�ru oldu�undan emin olun

# Kamera kayna��n� a� (0 varsay�lan kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera a��lamad�!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare al�namad�, ��k�l�yor...")
        break

    # YOLO modelini kullanarak tahmin yap
    results = model(frame)

    # Sonu�lar� g�r�nt� �zerine �iz
    annotated_frame = results[0].plot()

    # G�r�nt�y� g�ster
    cv2.imshow('YOLO Detection', annotated_frame)

    # 'q' tu�una bas�ld���nda ��k
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynaklar� serbest b�rak
cap.release()
cv2.destroyAllWindows()
