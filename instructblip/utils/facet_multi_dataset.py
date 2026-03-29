import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FacetMultiDataset(Dataset):
    def __init__(self, json_data, image_dir):

        self.data = json_data
        self.image_dir = image_dir
        

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_path = os.path.join(self.image_dir, item['filename'])
        image = Image.open(img_path).convert('RGB')
        
        
        return item['class1'], item['gender_presentation_masc'], item['gender_presentation_fem'], item['skin_tone'], image