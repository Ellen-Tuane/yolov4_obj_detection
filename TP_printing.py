from iou import previsao_TP
import cv2
import matplotlib.pyplot as plt

img2 = cv2.imread('/home/ellentuane/Documents/IC/image/Aerial_City_0.jpg')


for i in previsao_TP:
    xx, yy, ww, hh = i[0][0], i[0][1], i[0][2], i[0][3]
    cv2.rectangle(img2, (xx, yy), (ww, hh), (0, 255, 0), 2)

    xxx, yyy, www, hhh = i[1][0], i[1][1], i[1][2], i[1][3]
    cv2.rectangle(img2, (xxx, yyy), (www, hhh), (255, 0, 0), 2)

'''for j in tp_05:
    xxxx, yyyy, wwww, hhhh = j[0][0], j[0][1], j[0][2], j[0][3]
    cv2.rectangle(img2, (xxxx, yyyy), (wwww, hhhh), (0, 255, 255), 2)

    xxx1, yyy2, www3, hhh4 = j[1][0], j[1][1], j[1][2], j[1][3]
    cv2.rectangle(img2, (xxx1, yyy2), (www3, hhh4), (255, 255, 0), 3)'''


plt.imshow(img2)
plt.show()