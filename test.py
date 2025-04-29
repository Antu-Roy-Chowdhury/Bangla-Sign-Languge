import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
import math

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Enable detection of up to 2 hands

# Load the classifier model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Image size for resizing
imgSize = 300

# List of labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'o', 'a', 'i', 'u', 'e', 'oo', 'k', 'kh', 'g', 'gh', 'c', 'ch', 'j', 'jh', 't', 'th', 'd', 'dh', 'to', 'tho', 'do', 'dho', 'n', 'p', 'f', 'b', 'v', 'm', 'y', 'r', 's', 'h', 'rh', 'ng', 'bisrg']

while True:
    success, img = cap.read()
    if not success:
        continue

    # Flip image for mirror effect
    img = cv2.flip(img, 1)

    # Detect hands
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = None

    if hands:
        if len(hands) == 1:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            if x2 > x1 and y2 > y1:
                imgCrop = img[y1:y2, x1:x2]

        elif len(hands) == 2:
            # Get bounding boxes for both hands
            hand1 = hands[0]
            hand2 = hands[1]
            x1_h1, y1_h1, w1_h1, h1_h1 = hand1['bbox']
            x1_h2, y1_h2, w1_h2, h1_h2 = hand2['bbox']

            # Calculate the combined bounding box with some extra padding
            padding = 20
            x_min = max(0, min(x1_h1, x1_h2) - padding)
            y_min = max(0, min(y1_h1, y1_h2) - padding)
            x_max = min(img.shape[1], max(x1_h1 + w1_h1, x1_h2 + w1_h2) + padding)
            y_max = min(img.shape[0], max(y1_h1 + h1_h1, y1_h2 + h1_h2) + padding)

            if x_max > x_min and y_max > y_min:
                imgCrop = img[y_min:y_max, x_min:x_max]

        if imgCrop is not None:
            crop_h, crop_w = imgCrop.shape[:2]
            aspect_ratio = crop_h / crop_w

            if aspect_ratio > 1:
                k = imgSize / crop_h
                new_w = int(k * crop_w)
                resized = cv2.resize(imgCrop, (new_w, imgSize))
                w_gap = (imgSize - new_w) // 2
                imgWhite[:, w_gap:w_gap + new_w] = resized
            else:
                k = imgSize / crop_w
                new_h = int(k * crop_h)
                resized = cv2.resize(imgCrop, (imgSize, new_h))
                h_gap = (imgSize - new_h) // 2
                imgWhite[h_gap:h_gap + new_h, :] = resized

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predicted_label = labels[index]

            # Modern overlay with predicted label
            if hands:
                # Use the bounding box of the first detected hand for single hand,
                # or the combined bounding box for two hands
                if len(hands) == 1:
                    x_overlay, y_overlay, w_overlay, h_overlay = hands[0]['bbox']
                else:
                    x_overlay, y_overlay = x_min, y_min
                    w_overlay, h_overlay = x_max - x_min, y_max - y_min

                cv2.rectangle(img, (x_overlay - 20, y_overlay - 60), (x_overlay + w_overlay + 20, y_overlay - 10), (0, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{predicted_label}', (x_overlay, y_overlay - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.imshow("ImageWhite", imgWhite)

    # Draw main UI frame
    cv2.rectangle(img, (0, 0), (640, 40), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, "Sign Language Recognition", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()