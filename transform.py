
# coding: utf-8

# In[17]:


from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import math


def onmouse(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

def Get_Outline(input_dir):
    image = cv2.imread(input_dir)
    limit = 600
    if image.shape[0] > limit:
        print(image.shape)
        image = cv2.resize(image, (limit, int(limit*image.shape[0]/image.shape[1])))
        print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #edged = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=[0,0,0])
    #cv2.imshow("edged", edged)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing", closing)
    
    blurred = cv2.GaussianBlur(closing, (5,5),0)
    cv2.imshow("blurred", blurred)
    
    edged = cv2.Canny(blurred,75,200)
    cv2.imshow("edged", edged)
    print("get outline")
    return image,gray,edged

def getLen2(v1, v2):
    return (v1[0] - v2[0])*(v1[0] - v2[0]) + (v1[1] - v2[1])*(v1[1] - v2[1])

def getCross(kb1, kb2):
    x = (kb2[1] - kb1[1]) / (kb1[0] - kb2[0])
    y = kb1[0] * x + kb1[1]
    return np.array([x,y])

def addEdge(head, tail, edges):
    if head[0] > tail[0]:
        temp = head
        head = tail 
        tail = temp
    delta = 10
    for i in range(edges.shape[0]): 
        if (abs(head[0] - edges[i,0,0]) <= delta) and (abs(head[1] - edges[i,0,1]) <= delta) and (abs(tail[0] - edges[i,1,0]) <= delta) and (abs(tail[1] - edges[i,1,1]) <= delta):
            return edges
    edges = np.concatenate((edges, np.array([[head, tail]])), axis=0)
    return edges

def Get_cnt(edged, image):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if  imutils.is_cv2()  else   cnts[1]
    cnts =sorted(cnts,key=lambda c: cv2.arcLength(c,True),reverse=True)# 轮廓按大小降序排序
    print("len",len(cnts))
    
    #for i in range(len(cnts)):
        
        #创建白色幕布
        #temp = np.ones(edged.shape,np.uint8)*255
        #画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
        #cv2.drawContours(temp,cnts[i],-1,(0,255,0),3)
        #cv2.imshow("contours"+str(i),temp)
    
    edges = np.zeros((0, 2, 2))
    for i in range(0, min(4,len(cnts))):
        c = cnts[i]
        peri = cv2.arcLength(c,True)
        # approx = Get_corners(c)  # 获取近似的轮廓
        approx = cv2.approxPolyDP(c,0.02 * peri,True)
        #创建白色幕布
        temp = np.ones(edged.shape,np.uint8)*255
        #画出轮廓：temp是白色幕布，contours是
        temp = np.ones(edged.shape,np.uint8)*255
        #画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
        cv2.drawContours(temp,approx,-1,(0,255,0),3)
        cv2.imshow("approx"+str(i),temp)
        print("approx", approx)
        approx = approx.reshape((-1, 2))
        
        for i in range(1, approx.shape[0]):
            print("shape",np.array([[approx[i], approx[i-1]]]).shape)
            edges = addEdge(approx[i], approx[i-1], edges)
            #edges = np.concatenate((edges, np.array([[approx[i], approx[i-1]]])), axis=0)
        if approx.shape[0] >= 4:
            edges = addEdge(approx[approx.shape[0]-1], approx[0], edges)
            #edges = np.concatenate((edges, np.array([[approx[approx.shape[0]-1], approx[0]]])), axis=0)
    print("edges", edges.shape)
    print(edges)
    edges =sorted(edges,key=lambda edge: getLen2(edge[0], edge[1]),reverse=True)# 轮廓按大小降序排序
    
    edges = np.array(edges)
    print(edges.shape)
    kb = np.zeros((4, 2))
    for i in range(0, 4):
        if edges[i,1,0]-edges[i,1,1] == 0:
            kb[i,:]=np.array([np.inf, edges[i,1,0]])
        else:
            k = (edges[i,0,1]-edges[i,1,1])/(edges[i,0,0]-edges[i,1,0])
            kb[i,:] = np.array([k, edges[i,1,1]-k*edges[i,1,0]])
        print(edges[i],kb[i])
    pivot = 0
    while kb[pivot,0] == np.inf:
        pivot = pivot + 1
    idx = -1
    minDif = np.inf;
    for i in range(4):
        if i == pivot:
            continue
        dif = np.abs(kb[pivot, 0] - kb[i, 0])
        if dif != np.nan and dif < minDif:
            minDif = dif
            idx = i
    print(pivot, idx)
    
    docCnt = np.zeros((4, 2))
    num = 0
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    for i in range(0, 4):
        if i != pivot and i != idx:
            docCnt[num,:] = getCross(kb[pivot], kb[i])
            xmax = max(docCnt[num, 0], xmax)
            xmin = min(docCnt[num, 0], xmin)
            ymax = max(docCnt[num, 1], ymax)
            ymin = min(docCnt[num, 1], ymin)
            num = num +1
    for i in range(0, 4):
        if i != pivot and i != idx:
            docCnt[num,:] = getCross(kb[idx], kb[i])
            xmax = max(docCnt[num, 0], xmax)
            xmin = min(docCnt[num, 0], xmin)
            ymax = max(docCnt[num, 1], ymax)
            ymin = min(docCnt[num, 1], ymin)
            num = num +1
    print(docCnt)
    
    top = 0
    bottom = 0
    left = 0
    right = 0
    if ymin < 0:
        top = -ymin
    if ymax > image.shape[0]:
        bottom = ymax - image.shape[0]
    if xmin < 0:
        left = -xmin
    if xmax > image.shape[1]:
        right = xmax - image.shape[1]
    for i in range(4):
        docCnt[i, 0] = docCnt[i, 0] + left
        docCnt[i, 1] = docCnt[i, 1] + top
    
    expanded = cv2.copyMakeBorder(image, int(top), int(bottom), int(left), int(right), cv2.BORDER_REPLICATE)
    cv2.imshow("expanded", expanded)
    
    print("get cnt")
    
    return docCnt, expanded

if __name__=="__main__":
    input_dir = "12.jpg"
    image,gray,edged = Get_Outline(input_dir)
    docCnt, expanded = Get_cnt(edged, image)
    result_img = four_point_transform(expanded, docCnt.reshape(4, 2)) # 对原始图像进行四点透视变换
    cv2.imshow("original", image)
    #cv2.imshow("gray", gray)
    #cv2.imshow("edged", edged)
    cv2.imshow("result_img", result_img)
#    cv2.imwrite("result_" + input_dir, result_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()