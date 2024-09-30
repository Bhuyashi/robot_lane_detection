import cv2
import numpy as np
import matplotlib.pyplot as plt

from util import plotImage,hsvSlider

class preprocessImage():
    def __init__(self,img):
        self.img = img
        self.height, self.width, self.channel = img.shape

    def resize(self,w):
        h = int(w*(self.height/self.width))
        resized_img = cv2.resize(self.img,(w,h),interpolation=cv2.INTER_AREA)
        return preprocessImage(resized_img)
    
    def toRGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
    
    def toHSV(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

class coneDetection():
    def __init__(self,img):
        # Obtain range of HSV values for mask from HSV slider
        self.lower_red1 = np.array([0, 100, 170])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 150, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.img = img
        self.x_center = img.shape[1]//2
    
    def getMaskandOverlay(self):
        mask1 = cv2.inRange(self.img, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(self.img, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2
        overlay = cv2.bitwise_and(self.img, self.img, mask=mask)
        return mask, overlay

    def getConeCenters(self,mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for contour in contours:
            if cv2.contourArea(contour) > 1:  
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if cx <= self.x_center:
                        left_x.append(cx)
                        left_y.append(cy)
                    else:
                        right_x.append(cx)
                        right_y.append(cy)
        return left_x,left_y,right_x,right_y

def drawLines(img,lx,ly,rx,ry):
    ml,bl = np.polyfit(lx,ly,1)
    mr,br = np.polyfit(rx,ry,1)

    H = img.shape[1]
    x0 = int(-bl/ml)
    x1 = int((H-bl)/ml)
    x2 = int(-br/mr)
    x3 = int((H-br)/mr)
    cv2.line(img, (x0,0), (x1,H-1), (255, 0, 0))
    cv2.line(img, (x2,0), (x3,H-1), (255, 0, 0))
    return img

def saveImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('answer.png',img)

if __name__ == '__main__':
    img = cv2.imread('red.png')
    # Resize RAW image to a smaller dimension
    # Convert to RGB to be viewed in matplotlib
    img = preprocessImage(img).resize(300).toRGB()
    # Convert image to HSV domain to extract red hue
    hsv = preprocessImage(img).toHSV()
    # hsvSlider(img,hsv)    # Use slider to get HSV values for mask
    mask, overlay = coneDetection(hsv).getMaskandOverlay()
    fig, ax1,ax2,ax3,ax4 = plotImage([img,hsv,mask,overlay],['Image','HSV','Mask','Overlay'],4)
    lx,ly,rx,ry = coneDetection(hsv).getConeCenters(mask)
    final = drawLines(img.copy(),lx,ly,rx,ry)
    fig, ax1, ax2, ax3 = plotImage([img,mask,final],['Image','Mask','Final'],3)
    plt.show()
    saveImage(final)