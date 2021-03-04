#openCV - a popular lib of computer vision
#- uses cascades (divides the problem of face recog into several stages. each stage, rough and quick test takes place, where overall of 50 stages approx will be there and the face will be detected only if all stages pass the test.)
import cv2
import sys

imagePath = sys.argv[1] # enter the path of the image for which the face should be recognized
cascPath = "haarcascade_frontalface_default.xml" #enter the path of the casade or load the cascade
#load pretrained cascade
# Haar Cascade is a machine learning object detection algorithm 
#used to identify objects in an image or video...xml file 

# Haar Cascade is a machine learning-based approach where a lot of positive and negative images are used to train the classifier. 
#Positive images – These images contain the images which we want our classifier to identify. 
#Negative Images – Images of everything else, which do not contain the object we want to detect.


cascPath_eye = "haarcascade_eye.xml"  #path of cascade file to detect eyes



faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier(cascPath_eye)



#load our image
image = cv2.imread(imagePath)


#cv2.cvtColor() method is used to convert an image 
#from one color space to another
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#detectMultiScale function to detect face/obj's
#this algo uses moving window approach to find obj's.
#scale factor - compensates the faces appering zoom(due to the closeness to the camera) and far in the pic
#minNeighbors - defines how many objects are detected near the current one before it declares the face found.
#minSize - size of each window 



faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE) #returns list of all rectangles,which it detects a face

print("Found {0} faces!".format(len(faces)))

# If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 

# # Draw a rectangle around the faces
for (x, y, w, h) in faces: # x,y is rectangle's pos. w,h is rectangle’s width and height
	image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	#once we get this location of rectangles, ROI (Rectangle Region of Interest)
	#for face and eyes can be detected too.
	roi_grey = gray[y:y+h, x:x+w]
	roi_color = image[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_grey)
	for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(roi_color, (ex, ey),(ex+ew, ey+eh),(0,255,0), 2)


# #draw a rect using the above x,y,w,h values
cv2.imshow("Faces found with eyes detected", image) #display image with boundary detecting faces
cv2.waitKey(0)#wait for user to press any key
cv2.destroyAllWindows()