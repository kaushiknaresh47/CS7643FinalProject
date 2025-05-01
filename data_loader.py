import pandas as pd
import numpy as np
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import torchvision
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

class DataLoader():
    def __init__(self, size, use_fraction=0.5):
        self.use_fraction = use_fraction
        self.parquet_data = None
        self.labels = None
        self.size = size
        self.images = None
        self.labelNames = {0: 'blues'
                ,1: 'classical'
                ,2: 'country'
                ,3: 'deathmetal'
                ,4: 'doommetal'
                ,5: 'drumnbass'
                ,6: 'electronic'
                ,7: 'folk'
                ,8: 'grime'
                ,9: 'heavymetal'
                ,10: 'hiphop'
                ,11: 'jazz'
                ,12: 'lofi'
                ,13: 'pop'
                ,14: 'psychedelicrock'
                ,15: 'punk'
                ,16: 'reggae'
                ,17: 'rock'
                ,18: 'soul'
                ,19: 'techno'
               }

        self.train_images = None
        self.train_labels = None

        self.test_images = None
        self.test_labels = None

    def download_data(self):
        print("Downloading data...")
        self.parquet_data = pd.read_parquet("hf://datasets/eong/20k-Album-Covers-within-20-Genres/data/train-00000-of-00001-f37f5042abc5be8d.parquet")

    def test_download(self):
        print("testing download")
        # Extract the first image from the DataFrame
        image_data = self.parquet_data.loc[12, 'image']['bytes']  # Adjust index and column name as needed

        # Convert bytes to image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Show or save the image
        plt.imshow(image)
        plt.show()


        to_tensor = T.Compose([
            T.Resize((self.size, self.size)),  
            T.ToTensor(),          
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        # to_tensor = torchvision.transforms.ToTensor()
        image_tensor = to_tensor(image)  # shape: [3, H, W]

        print(image_tensor.shape)
        print(image_tensor)

    def process_to_tensors(self, verbose=False):
        print("converting data to tensors")
        # loop to extract the parquet data into two lists: one of labels and one of tensors
        labels = torch.ones((20000, 1))
        im_tensors = torch.empty((20000, 3, self.size, self.size))
        to_tensor = T.Compose([
            T.Resize((self.size, self.size)),  
            T.ToTensor(),          
            # T.Normalize(
            #     mean=[0.485, 0.456, 0.406], 
            #     std=[0.229, 0.224, 0.225]
            # )
        ])
        # to_tensor = torchvision.transforms.ToTensor()
        for i in range(len(self.parquet_data)):
            if i % 1000 == 0:
                print(i)
            labels[i] = self.parquet_data.loc[i, 'label']
            im_bytes = self.parquet_data.loc[i, 'image']['bytes']
            image = Image.open(io.BytesIO(im_bytes))
            image_tensor = to_tensor(image)
            im_tensors[i] = image_tensor
            if (i == 0 or i == 19999) and verbose:
                plt.imshow(image)
                plt.show()
                plt.title(self.labelNames[labels[i]])
        # self.labels = labels
        # self.images = im_tensors
        n_data = labels.shape[0]
        subset_size = int(self.use_fraction * n_data)

        # Convert to numpy for sklearn compatibility
        labels_np = labels.cpu().numpy()

        # Stratified sampling: get 50% with preserved class proportions
        if self.use_fraction<1:
            _, selected_indices = train_test_split(
                range(n_data),
                train_size=subset_size,
                stratify=labels_np,
                random_state=0
            )
        else:
            selected_indices = range(n_data)
        print('n_indices selected', len(selected_indices))
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)

        # Subset the data
        self.images = im_tensors[selected_indices]
        self.labels = labels[selected_indices]

    def verify_process(self, im_ind = None):
        print("verifying processing")
        # verifying images were correctly transformed...
        if  im_ind is None:
            im_ind = 8908
            image_tensor = self.images[8908]  #  test image, album artwork has a green goblin
        else:
            
            image_tensor =  self.images[im_ind]  # 0 should be lumineers, 19999 should be slam
        to_pil = torchvision.transforms.ToPILImage()
        image = to_pil(image_tensor)
        plt.imshow(image)
        plt.title(f"label = {self.labelNames[self.labels[im_ind].long().numpy()[0]]}")
        plt.show()

    def create_train_test(self):
        nData = self.labels.shape[0]
        torch.manual_seed(0)
        shuffle = torch.randperm(nData)
        train = shuffle[np.array(range(0, int(0.8 * nData)))]
        test = shuffle[np.array(range(int(0.8 * nData), nData))]

        self.train_labels = self.labels[train]
        self.train_images = self.images[train]
        self.test_labels = self.labels[test]
        self.test_images = self.images[test]

    def return_train_data(self):
        return self.train_images, self.train_labels

    def return_test_data(self):
        return self.test_images, self.test_labels

    def main(self):
        self.download_data()
        self.test_download()
        self.process_to_tensors(verbose=False)
        self.verify_process()
        self.create_train_test()

if __name__ == "__main__":
    dl = DataLoader(size=224)
    dl.main()