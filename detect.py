import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load("vegetable_model.pth"))
model.eval()

classes = ["potato", "tomato", "cucumber"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output,1)

    label = classes[predicted.item()]

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Smart Scan AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
