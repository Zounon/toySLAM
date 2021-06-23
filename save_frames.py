import cv2 
cap = cv2.VideoCapture('test_countryroad.mp4')
success, image = cap.read()
count = 0
while success: 
    cv2.imwrite("frame%d.jpg" % count, image)
    success, image = cap.read()
    print('sdf')
    count += 1