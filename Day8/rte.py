import cv2


import urllib.request
import numpy as np

# Define the URL
url = 'http://192.168.31.186/cam-lo.jpg'
# '''cam.bmp / cam-lo.jpg / cam-hi.jpg / cam.mjpeg'''

cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

while True:
    # Fetch image from URL
    img_resp = urllib.request.urlopen(url)  # Only use 'url' here
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)
    (cnt, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    k = img.copy()
    cv2.drawContours(k, cnt, -1, (0, 255, 0), 2)

    # Show images
    cv2.imshow("mit contour", canny)
    cv2.imshow("live transmission", img)

    # Check for 'q' to quit or 'a' to print contour count
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('a'):
        cow = len(cnt)
        print("Number of contours:", cow)
        if cow>20:
            urllib.request.urlopen('http://192.168.31.186/led-on')
        else:
            urllib.request.urlopen('http://192.168.31.186/led-off')


cv2.destroyAllWindows()