import cv2
import time


#cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)



if not cap.isOpened():
    print("Error: Camera not found")
    exit()

start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera (1 second)", frame)

    # Close after 1 second OR if user presses 'q'
    if time.time() - start >= 6 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
