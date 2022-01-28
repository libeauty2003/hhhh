import cv2 as cv
src=cv.imread('20211202141735.png')
gray=cv.cvtColor(src,cv.COLOR_BGR2GRAY)
cv.imwrite('fff.png',gray)
cv.imshow('flower',gray)
cv.waitKey(0)
cv.destroyWindow()

