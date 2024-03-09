from torchfusion_utils.fp16 import convertToFP16
from PIL import Image
from torchfusion_utils.initializers import *
from torchfusion_utils.metrics import Accuracy
from torchfusion_utils.models import load_model,save_model
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.autograd import Variable
import cv2
import collections
#import google.colab
#from google.colab.patches import cv2_imshow
#from fire.ipynb import *
from PIL import Image
import glob
import os
import streamlit as st
import collections
from streamlit_option_menu import option_menu
from pushbullet import Pushbullet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.title("System Page")

def load_image(image_up):
    img = Image.open(image_up)
    return img

# Main Streamlit app
if __name__ == "__main__":
    st.title("Image Uploader App")

image_up = st.file_uploader("Upload A Picture Here",type=['png','jpg,','jpeg'])

if image_up is not None:
        st.image(image_up)
        file_details = {"FileName": image_up.name, "FileType": image_up.type}

        # Save the uploaded image
        with open(os.path.join("tempDir", image_up.name), "wb") as f:
            f.write(image_up.getbuffer())

if image_up is not None:
    file_details = {"FileName":image_up.name,"FileType":image_up.type}
    img = load_image(image_up)
    with open(os.path.join("tempDir",image_up.name),"wb") as f: 
      f.write(image_up.getbuffer())         
    st.success("Image Saved")

model = torch.load("./fire-flame.pt")

load_saved_model = torch.load('fire-flame.pt')

dummy_input = torch.FloatTensor(1,3,224,224)
dummy_input = dummy_input.to(device)

torch.onnx.export(load_saved_model, dummy_input, 'fire-flame.onnx')

transformer = transforms.Compose([transforms.Resize(225),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])])


img = plt.imread(os.path.join("tempDir", image_up.name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img.astype('uint8'))
orig = img.copy()
img_processed = transformer(img).unsqueeze(0)
img_var = Variable(img_processed, requires_grad= False)
img_var = img_var.cuda()
load_saved_model.eval()
logp = load_saved_model(img_var)
expp = torch.softmax(logp, dim=1)
confidence, clas = expp.topk(1, dim=1)
co = confidence.item() * 100
class_no = str(clas).split(',')[0]
class_no = class_no.split('(')
class_no = class_no[1].rstrip(']]')
class_no = class_no.lstrip('[[')

orig = np.array(orig)
orig = cv2.cvtColor(orig,cv2.COLOR_BGR2RGB)
orig = cv2.resize(orig,(800,500))

if class_no == '1':
    label = "Nuetral: " + str(co)+"%"
    cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    st.write(label)
  
elif class_no =='2':
    label = "Smoke: " + str(co)+"%"
    cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    st.write(label)

elif class_no == '0':
    label = "Fire: " + str(co)+"%"
    cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    st.write(label)

output_path = os.path.join("/content/output", image_up.name)
cv2.imwrite(output_path, orig)

st.header("ALERT AUTHORITY HERE")

fireButton = st.button("Fire Button")
smokeButton = st.button("Smoke Button")

API_KEY = "o.sKpoGn1um8K7f4r0Xn02B5VreRcquuKo"
file1 = "fire.txt"
file2 = "smoke.txt"

if fireButton:
  with open(file1,mode = 'r') as fire:
    fire = fire.read()
    pb = Pushbullet(API_KEY)
    push = pb.push_note('ALERT',fire)

elif smokeButton:
  with open(file2,mode = 'r') as smoke:
    smoke = smoke.read()
    pb = Pushbullet(API_KEY)
    push = pb.push_note('ALERT',smoke)