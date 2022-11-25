import cv2
img = cv2.imread("/Users/aryansingh/projects/image_segmentation/chest_xray/test/PNEUMONIA/person1_virus_12.jpeg")

height, width, channels = img.shape
crop_img = img[0:height, 10:width-10]
#crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)


