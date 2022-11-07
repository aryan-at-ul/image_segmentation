import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision.io as io
import numpy as np
from IPython.display import Image as dImage
import os
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle


model = models.resnet18(pretrained=True)

feature_extractor = create_feature_extractor(
	model, return_nodes=['avgpool'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fet_from_img(img):
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]

   # print(img.shape)

   mean = [0.485, 0.485, 0.485]
   std = [0.229, 0.229, 0.229]

   transform_norm = transforms.Compose([transforms.ToTensor(),
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      return out['avgpool']#.cpu().detach().numpy().ravel()



def fet_img_image(image_path,model):
   img = Image.open(image_path)
   mean = [0.485, 0.456, 0.406]
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.ToTensor(),
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to(device)
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      out = feature_extractor(img_normalized)
      return out.cpu().detach().numpy().ravel()


if __name__ == "__main__":
    print("ok")
