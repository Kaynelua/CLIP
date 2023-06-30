import torch
import clip
from PIL import Image
from torchreid import distance
from torchreid import reidtools

import os
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B-32.pt", device=device) #clip.load("ViT-B/32", device=device)

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#print(preprocess(Image.open("CLIP.png")).size())
rootDir = "data/places365_large/test/gallery" 
firstIteration = True
for dirs in sorted(os.listdir(rootDir)):
    dirPath = os.path.join(rootDir,dirs)
    for fileName in sorted(os.listdir(dirPath)):
        imagePath = os.path.join(dirPath,fileName)
        print("Processing:",imagePath)
        input_img = preprocess(Image.open(imagePath)).unsqueeze(0)
        if firstIteration:
            imageMatrix = input_img
            firstIteration = False
        else:
            imageMatrix = torch.cat((imageMatrix,input_img),dim=0)
imageMatrix = imageMatrix.to(device)

#image_array = torch.stack( (preprocess(Image.open("CLIP.png")),preprocess(Image.open("image059.jpg"))) ).to(device)
#print(image_array.size())

queryList = ["image of forest", "image of short buildings", "image of tall buildings", "image of a coast", "image of a conference room", "People walking on the beach"]
text = clip.tokenize(queryList).to(device)

with torch.no_grad():
    print(imageMatrix.size())
    image_features = model.encode_image(imageMatrix) #image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    distMatrix = distance.compute_distance_matrix(text_features,image_features,metric="cosine")
    reidtools.visualize_ranked_results_text(distmat=distMatrix.to("cpu").numpy(),queryList = queryList)
    #sortedMatrix,indices = torch.sort(distMatrix,dim=0)
    #for idx,orderedIndices in enumerate(indices) :
    #    print("Row:" , idx, orderedIndices[:10])

    
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)
