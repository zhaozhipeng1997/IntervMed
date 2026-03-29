import csv
import json

filepath = "your path/FACET_Dataset/annotations/annotations.csv"
output_file = "./pifu_zhiye_gender.json"  

processed_data = []

with open(filepath, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        tone_values = {
            i: int(row[f"skin_tone_{i}"]) 
            for i in range(1, 11)
        }
        
        max_value = max(tone_values.values())
        
        if max_value == 0:
            max_skin_no = 0
            continue
        else:
            max_indices = [i for i, v in tone_values.items() if v == max_value]
            max_skin_no = max_indices[0]

        data_entry = {
            "filename": row["filename"],
            "class1": row["class1"],
            "gender_presentation_masc": row["gender_presentation_masc"],
            "gender_presentation_fem": row["gender_presentation_fem"],
            "skin_tone": max_skin_no
        }
        
        processed_data.append(data_entry)

with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

print(f"Processing complete! Total records processed: {len(processed_data)}")
print(f"Results saved to: {output_file}")
