import cv2
import numpy as np

### LOADING YOLO

net = cv2.dnn.readNet("weights/yolov3.weights", "configs/yolov3.cfg")
classes = []

with open('coco.names.txt', 'r') as f:
    classes = f.read().splitlines()

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


### LOADING IMAGES ###

img = cv2.imread("image.jpg")
img = cv2.resize(img, None, fx=1.5, fy=1.5)
height, width, channels = img.shape

### DETECTING OBJECTS ### 
blob = cv2.dnn.blobFromImage(img, 1/255, (608,608), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


### SHOWING INFORMATION ON THE SCREEN ###
boxes=[]
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            
            ### DETECTING OBJECTS ### 

            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            
            ### RECTANGLE CORDINATES ###
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            print([x,y,w,h])

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

#print(len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
print("Number of things detected: ", len(indexes))

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    for j in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+30), font, 3, color, 3)


cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



