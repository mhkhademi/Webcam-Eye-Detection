import cv2 as cv
myCamera = cv.VideoCapture(0)
eyeCascade = cv.CascadeClassifier("Face Detect\\haarcascades\\haarcascade_eye.xml")

while True:
   check,frame = myCamera.read()
   if check:
      eye = eyeCascade.detectMultiScale(frame)#,scaleFactor=1.01,minNeighbors = 50)

      if len(eye)>0:
         for (x,y,w,h) in  eye:
             our_image_rect = cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
         cv.imshow("Eye detect",our_image_rect)

   key = cv.waitKey(1)
   if key == ord('q'):
      break
   
myCamera.release()
cv.destroyAllWindows()


