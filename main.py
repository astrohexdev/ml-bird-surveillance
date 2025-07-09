from utils.prediction import predict
import cv2

cap = cv2.VideoCapture('video/tow_1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        print('check path')
        break

    predict(frame)
    cv2.namedWindow("Bird Species", cv2.WINDOW_NORMAL)
    cv2.imshow('Bird Species', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
