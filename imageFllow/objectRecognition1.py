
import cv2

cap = cv2.VideoCapture('./videos/cars.mp4')
background = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while(1):
    ret, frame = cap.read()
    backgroundMask = background.apply(frame)
    median = cv2.medianBlur(backgroundMask, 3)

    (contours, hierarchy) = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #/Burada 500'den küçük olan cisimleri algılamayacak ve kare cizimler gerçekleşecek .
    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    backgroundSize = cv2.resize(backgroundMask, (600, 360))

    frame1 = cv2.resize(frame, (600, 360))

    cv2.imshow('cameraRecord', backgroundSize)
    cv2.imshow('cameraRecord1', frame1)

    k = cv2.waitKey(1) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()

print("Finish...")
