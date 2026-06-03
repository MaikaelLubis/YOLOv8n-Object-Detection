import cv2
from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")

cam = cv2.VideoCapture(0)

print("Menghubungkan ke kamera... Tekan tombol 'q' untuk keluar.")

while cam.isOpened():
    success, frame = cam.read()
    
    if not success:
        print("Gagal membuka kamera Laptop.")
        break

    results = model(frame, imgsz=540, stream=True)

    # Gambar kotak (bounding box) dan label hasil deteksi ke layar
    for r in results:
        annotated_frame = r.plot()

    # Tampilkan hasil kamera yang sudah ada kotak deteksinya
    cv2.imshow("PBL-RE-082 - YOLOv8 Real-Time Detection", annotated_frame)

    # Program akan berhenti jika kamu menekan tombol 'q' di keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup kamera setelah selesai
cam.release()
cv2.destroyAllWindows()