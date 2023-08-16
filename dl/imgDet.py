import cv2 as cv
import torch
from troch.autograd import Variable

frame = cv.imread("../data/dog.jpg")
frame = cv.resize(frame, (224, 224))
frame.reshape(1,3,224,224)
frame = Variable(torch.from_numpy(frame).float())
net = torch.hub.load("pytorch/vision", " ", weights="DEFAULT")

out = net(frame)
detections = out.data
scale = torch.Tensor([frame.shape[3], frame.shape[2],
                       frame.shape[3], frame.shape[2]])
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        pt = (detections[0,i,j,1:]*scale).numpy()
        cv.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
        j += 1

cv.imshow("showPic", frame)
cv.waitKey(0)
cv.destroyAllWindows()
