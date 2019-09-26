import cv2
img = cv2.imread("000002281.jpeg")
cv2.rectangle(img, (59,335), (59+112,335+204),(255,0,0),2)
cv2.imshow("image",img)
cv2.waitKey(0)