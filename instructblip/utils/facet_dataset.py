import csv
import json
from tqdm import tqdm
import os
from PIL import Image

class FacetDataset:
    def __init__(self, annotation_path: str, image_root: str, max_samples=50):
        self.image_root = image_root
        self.max_samples = max_samples
        self.data = []
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print(row['person_id'], row['filename'], row['class1'], row['gender_presentation_masc'], row['gender_presentation_fem'])
                processed = {
                    'person_id': row['person_id'],
                    'filename': row['filename'],
                    'class1': row['class1'],
                    'gender_presentation_masc': row['gender_presentation_masc'],
                    'gender_presentation_fem': row['gender_presentation_fem']
                }
                self.data.append(processed)
        
        self._create_subset()

    def _create_subset(self):
        with open('utils/zhiye_gender.json', 'r') as f:
            target_combinations = json.load(f)
        
        self.subset = []
        total=self.max_samples*len(target_combinations)
        progress = tqdm(total=total, desc="Filter Samples")
        counters = [0] * len(target_combinations)
        
        for idx, combo in enumerate(target_combinations):
            for item in self.data:
                if (item.get('gender_presentation_masc')==combo.get('gender_presentation_masc') and 
                item.get('gender_presentation_fem')==combo.get('gender_presentation_fem') and
                item.get('class1') == combo.get('class1')):
                    self.subset.append(item)
                    counters[idx] += 1  
                    if counters[idx] >=self.max_samples:
                        break
                    progress.update(1)
        progress.close()
        

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        item = self.subset[idx]
        image_path = os.path.join(self.image_root, item['filename'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Unable to load image {image_path}: {str(e)}")
            return None, "", ""
        return image, item['class1'], item['gender_presentation_masc'], item['gender_presentation_fem']
