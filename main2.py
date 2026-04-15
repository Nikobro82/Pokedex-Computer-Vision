import torch
import cv2
from torchvision import models
from torchvision import transforms as transform
from torch.utils.data import DataLoader, Dataset

import os
from PIL import Image

import torch.optim as optim
import time

import numpy as np

from torch.nn.functional import cosine_similarity




Pokemons = []
PokemonsFound = {}

rag_database = []

def get_model():
    weights = None
    weights = models.ResNet50_Weights.DEFAULT


    model = models.resnet50(pretrained=True)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Identity()

    

    return model
model = get_model()

class UTKDataset:
    def __init__(self, model):
        self.Images = []
        self.Labels = []

        path = os.scandir("./Pokemon")

        image_num = 0
        max_images = 10600
        for filename in path:
            pokemon_name = filename.name

            folder = os.scandir(f"./Pokemon/{pokemon_name}")
            for filename2 in folder:
                with open(filename2.path, "r") as f:

                
                    if not pokemon_name in PokemonsFound:
                        PokemonsFound[pokemon_name] = len(Pokemons)
                        Pokemons.append(pokemon_name)

                    

                    img = cv2.imread(filename2.path)

                    if img is None:
                        continue

                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    transforms = transform.Compose([ 
                        transform.Resize((224,224)),
                        transform.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])

                    tensor = transform.ToTensor()(img_color)
                    video_tensor = transforms(tensor)

                    if video_tensor.size() != (3,224,224):
                        print("NOT RIGHT SIZE")
                        print(video_tensor.size())
                        continue 
                    
                    video_tensor = video_tensor.unsqueeze(0)
                    self.Images.append(video_tensor)

                    label = np.zeros(shape=(149))
                    
                    pokemon_index = PokemonsFound[pokemon_name]

                    label[pokemon_index] = 1


                    self.Labels.append(label)


                    print(f"Completion {(image_num/max_images) * 100:.2f}% : Pokemon: {Pokemons[pokemon_index]}")
                    image_num += 1

                    embed = model(video_tensor)

                    dict_appending = {"pokemon" : Pokemons[pokemon_index], "embedding" : embed}

                    rag_database.append(dict_appending)

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, key):
        tensor = self.Images[key]
        label = self.Labels[key]

        return tensor, label

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def get_similarities(input_tensor, embedding_matrix):
    input_tensor = model(input_tensor)
    output = cosine_similarity(input_tensor, embedding_matrix, dim=1)

    return torch.topk(output, k=3).indices

def get_prediction(input_tensor, embedding_matrix):
    similarities_indices = get_similarities(input_tensor, embedding_matrix).numpy()[0]
    
    stringUsing = ""


    for indice in similarities_indices:
        print(indice)
        pokemon_index = indice

        Pokemon = Pokemons[pokemon_index]
        stringUsing = f"{stringUsing}, {Pokemon}"
    
    print(stringUsing)


def get_embedding_matrix():
    embeds = []
    for dict_checking in rag_database:
        embeds.append(dict_checking["embedding"])

    return torch.stack(embeds)


def main():
    TRAIN_MODE = True
    if TRAIN_MODE == True:
        global rag_database
        global PokemonsFound
        global Pokemons

        dataset = UTKDataset(model)
        print("RAG DATABASE LOADED")
        torch.save(rag_database, "RAG_Database.pt")
        torch.save(Pokemons, "Pokemons.pt")
        torch.save(PokemonsFound, "PokemonsFound.pt")
    else:
        rag_database = torch.load("RAG_Database.pt")
        Pokemons = torch.load("Pokemons.pt")
        PokemonsFound = torch.load("PokemonsFound.pt")
    
    embedding_matrix = get_embedding_matrix()

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        if ret:
            video_tensor = transform.ToTensor()(frame)
            transforms = transform.Compose([
                transform.Resize(224)
            ])
            video_tensor = transforms(video_tensor)
            video_tensor = video_tensor.unsqueeze(0)
            get_prediction(video_tensor, embedding_matrix)
        
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("code finished executing")
    
main()



