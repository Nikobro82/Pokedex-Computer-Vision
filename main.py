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

Pokemons = []
PokemonsFound = {}

class UTKDataset:
    def __init__(self):
        self.Images = []
        self.Labels = []

        path = os.scandir("./Pokemon/dataset")

        image_num = 0
        max_images = 10600
        for filename in path:
            pokemon_name = filename.name

            folder = os.scandir(f"./Pokemon/dataset/{pokemon_name}")
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
                        transform.Resize((224,224))
                    ])

                    tensor = transform.ToTensor()(img_color)
                    video_tensor = transforms(tensor)

                    if video_tensor.size() != (3,224,224):
                        print("NOT RIGHT SIZE")
                        print(video_tensor.size())
                        continue 

                    self.Images.append(video_tensor)

                    label = np.zeros(shape=(149))
                    
                    pokemon_index = PokemonsFound[pokemon_name]

                    label[pokemon_index] = 1


                    self.Labels.append(label)


                    print(f"Completion {(image_num/max_images) * 100:.2f}% : Pokemon: {Pokemons[pokemon_index]}")
                    image_num += 1


                    
        
        '''with open(e.path, "r") as f:
                image_num += 1
                img = cv2.imread(e.path)
                gender_label = e.name.split("_")[1]

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                transforms = transform.Compose([
                    transform.Resize(224),
                ])

                tensor = transform.ToTensor()(img_rgb)

                new_image = transforms(tensor)

                print(f"Completion {(image_num/max_images) * 100:.2f}% : {new_image.size()}, Gender: {gender_label}")

                self.Images.append(tensor)
                self.Labels.append(int(gender_label))'''
            

            


    def __len__(self):
        return len(self.Images)

    def __getitem__(self, key):
        tensor = self.Images[key]
        label = self.Labels[key]

        return tensor, label

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def train(model, train_loader):
    epochs = 5

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1
    )
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"EPOCH: {epoch}")
        for batch_indx, (tensor, labels) in enumerate(train_loader):
            print(f"BATCH: {batch_indx}")
            
            labels = torch.tensor(labels)
            y_pred = model(tensor)
            loss = criterion(y_pred, labels)

            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model_weights.pth")

def get_model(TRAIN_MODE):
    weights = None
    if TRAIN_MODE:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = torch.load("model_weights.pth", weights_only=True)


    model = models.resnet50(weights=weights)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(2048, 149)

    for param in model.fc.parameters():
        param.requires_grad = True
    
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.layer3.parameters():
        param.requires_grad = True

    return model


def main():
    TRAIN_MODE = True
    dataset = UTKDataset()

    model = get_model(TRAIN_MODE)
    
    train_loader = DataLoader(dataset, batch_size=128, shuffle = True)



    if TRAIN_MODE:
        train(model, train_loader)

    


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
            pred = model(video_tensor)
            print(Pokemons[pred.argmax().item()])
        
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("code finished executing")



    

main()



