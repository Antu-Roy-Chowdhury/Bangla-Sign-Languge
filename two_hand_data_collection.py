import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector

offset = 10
imgSize = 300
counter = 0
folderPath = "Data"
folderPath = "Data/"
newFolderName = input("Enter the name for the new folder: ")
newFolderPath = os.path.join(folderPath, newFolderName)

if not os.path.exists(newFolderPath):
    os.makedirs(newFolderPath)
    print(f"Folder '{newFolderName}' created at '{newFolderPath}'")



try:
    numImagesToCollect = int(input("Enter the number of images to collect: "))
except ValueError:
    print("Invalid input. Please enter a number.")
    numImagesToCollect = 0

imagesCollected = 0

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands and len(hands) == 1:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset, :]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap, :] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :, :] = imgResize

        cv2.imshow("ImageWhite", imgWhite)

    elif hands and len(hands) == 2:  # Ensure two hands are detected
        hand1 = hands[0]
        hand2 = hands[1]

        # Get bounding boxes for both hands
        x1, y1, w1, h1 = hand1['bbox']
        x2, y2, w2, h2 = hand2['bbox']

        # Calculate the combined bounding box
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)

        # Crop the combined region
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset, :]
        imgCropShape = imgCrop.shape

        aspectRatio = (y_max - y_min) / (x_max - x_min)
        if aspectRatio > 1:
            k = imgSize / (y_max - y_min)
            wCal = math.ceil(k * (x_max - x_min))
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap, :] = imgResize
        else:
            k = imgSize / (x_max - x_min)
            hCal = math.ceil(k * (y_max - y_min))
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :, :] = imgResize

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    
    if imagesCollected >= numImagesToCollect:
        
        print("Image Collection Completed")
        break
    if key == ord('a'):
        imagesCollected += 1
        cv2.imwrite(f'{newFolderPath}/Image_{newFolderName}_{imagesCollected}.jpg', imgWhite)
        print("Image Saved :", imagesCollected)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()