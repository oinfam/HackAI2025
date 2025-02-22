import mouse
import time
import cv2

x = [0,99999]
y = [0,99999]


for i in x:
    for j in y:
        mouse.move(i,j, absolute = True, duration = .2)
        time.sleep(1)
        print(mouse.get_position())
       

'''
1. Get hand position
    a. 
2. Mouse move to hand position
3. Mouse click if left hand
'''