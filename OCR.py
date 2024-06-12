import os
import cv2
from scipy import ndimage
import numpy as np
import easyocr

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
PATH = "database/"

RESULTS = "result/"
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)

CH = "ready_to_detect/"
if not os.path.exists(CH):
    os.makedirs(CH)

def extract_num(path):
    for filename in os.listdir(path):
        img = cv2.imread(path+filename)
        
        if img is not None:
            #conver into gray
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            deNoised = ndimage.median_filter(gray_img, 3)
            #high pass filter
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            highPass = clahe.apply(deNoised)
            nplate = cascade.detectMultiScale(highPass,1.1,4)
            #crop function
            for (x,y,w,h) in nplate:
                wT,hT,cT=img.shape
                a,b=(int(0.02*wT),int(0.02*hT))
                plate=img[y+a:y+h-a,x+b:x+w-b,:]
                #make image more dark
                kernel=np.ones((1,1),np.uint8)
                plate=cv2.dilate(plate,kernel,iterations=1)
                plate=cv2.erode(plate,kernel,iterations=1)
                plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
                # (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
                cv2.imwrite(CH+filename,plate_gray)
                #reading the text
            reader = easyocr.Reader(['en'],gpu=False)
            text = reader.readtext(plate)
            for t in text:    
                bbox, text_,score = t
                print(text_)
                #showing the output
                cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
                cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
                cv2.putText(img,text_,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.imwrite(RESULTS+filename,img)

print("-----------Image Extration-----------")
print("-------------------------------------")
print("-----Initializing Image Extration----")
#calling 
extract_num(path=PATH)
print("------Image Extration completed------")

