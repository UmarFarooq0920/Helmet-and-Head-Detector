from PIL import Image
import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

image_path = 'hard_hat_workers241_png.rf.713a430de396801995becc01eeeeb9ef.jpg'
num_classes = 3
class_names = ['head','helmet','person']
session = onnxruntime.InferenceSession('best.onnx')

batch_size = session.get_inputs()[0].shape[0]
img_size_h = session.get_inputs()[0].shape[2]
img_size_w = session.get_inputs()[0].shape[3]

def letterbox_image(image, size):
    iw, ih = image.size
    #print(iw,ih)
    w, h = size
    scale = min(w / iw, h / ih)
    #print(scale)
    nw = int(iw * scale)
    nh = int(ih * scale)
    #print(nw,nh)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

image1 = Image.open(image_path)
resized = letterbox_image(image1, (img_size_w, img_size_h))

img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
img_in = np.expand_dims(img_in, axis=0)
img_in /= 255.0

input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: img_in})
#print(outputs)

cls = []
lble = []
boxs=[]
for i in outputs[0][0]:
  if i[4]>0.5:
    #print(np.argmax(i[5:]))
    cls.append(float(i[5:][np.argmax(i[5:])]))
    lble.append(np.argmax(i[5:]))
    x,y,w,h = i[0:4]
    boxs.append([x,y,w,h])
  
#print(len(boxs))
#print(len(cls))




boxes = []

for i in range(len(boxs)):
  x,y,w,h = boxs[i]
  x1 = x-0.5*w
  y1 = y-0.5*h
  x2 = x+0.5*w
  y2 = y+0.5*h
  boxes.append([int(x1),int(y1),int(x2),int(y2)])

nms_boxes = cv2.dnn.NMSBoxes(boxes, cls,score_threshold=0.4,nms_threshold=0.4,top_k=-1)
print(nms_boxes)


resized2 = np.array(resized)
for i in range(len(boxs)):
    if i in nms_boxes:
        x1,y1,x2,y2 = boxes[i]
        print(x1,y1,x2,y2)
        cv2.rectangle(resized2,(x1,y1),(x2,y2),(0,0, 0),1)
        cv2.putText(resized2, class_names[lble[i]], (int(x1), int(y1-5)),0,fontScale=2, color= (255, 0, 0),
                    thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('image',resized2)
cv2.waitKey(0) 
