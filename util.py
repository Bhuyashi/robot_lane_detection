import cv2
import numpy as np
import matplotlib.pyplot as plt

def plotImage(images,captions,n):
    fig, axes = plt.subplots(ncols=n)
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].set_title(captions[i])
    return fig, *axes,

def hsvSlider(img,hsv):
    cv2.namedWindow('HSV Mask Slider')

    def nothing(x):
        pass

    # Create trackbars for lower and upper ranges of Hue, Saturation, and Value
    cv2.createTrackbar('LH', 'HSV Mask Slider', 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar('LS', 'HSV Mask Slider', 0, 255, nothing)  # Lower Saturation
    cv2.createTrackbar('LV', 'HSV Mask Slider', 0, 255, nothing)  # Lower Value
    cv2.createTrackbar('UH', 'HSV Mask Slider', 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar('US', 'HSV Mask Slider', 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar('UV', 'HSV Mask Slider', 255, 255, nothing)  # Upper Value

    while True:
        # Get current positions of all trackbars for lower and upper HSV values
        l_h = cv2.getTrackbarPos('LH', 'HSV Mask Slider')
        l_s = cv2.getTrackbarPos('LS', 'HSV Mask Slider')
        l_v = cv2.getTrackbarPos('LV', 'HSV Mask Slider')
        
        u_h = cv2.getTrackbarPos('UH', 'HSV Mask Slider')
        u_s = cv2.getTrackbarPos('US', 'HSV Mask Slider')
        u_v = cv2.getTrackbarPos('UV', 'HSV Mask Slider')

        # Define lower and upper HSV bounds
        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        # Create a mask that identifies regions in the image within the selected HSV range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply the mask on the original image
        result = cv2.bitwise_and(img, img, mask=mask)

        # Show the original image, the mask, and the result
        cv2.imshow('Original Image', img)
        cv2.imshow('HSV Mask', mask)

        # Break the loop when 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()