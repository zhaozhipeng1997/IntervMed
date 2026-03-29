import csv
import json

filepath = "your path/FACET_Dataset/annotations/annotations.csv"
output_file = "./toufa_pifu_zhiye_gender.json"  

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
        hair_color_brown = row["hair_color_brown"]
        hair_color_blonde = row["hair_color_blonde"]
        hair_color_grey = row["hair_color_grey"]
        hair_color_black = row["hair_color_black"]
        hair_color_colored = row["hair_color_colored"]
        hair_color_red = row["hair_color_red"]

        hairtype_coily = row["hairtype_coily"]
        hairtype_dreadlocks = row["hairtype_dreadlocks"]
        hairtype_bald = row["hairtype_bald"]
        hairtype_straight = row["hairtype_straight"]
        hairtype_curly = row["hairtype_curly"]
        hairtype_wavy = row["hairtype_wavy"]
        hair_color_values = {
            'brown': hair_color_brown,
            'blonde': hair_color_blonde,
            'grey': hair_color_grey,
            'black': hair_color_black,
            'colored': hair_color_colored,
            'red': hair_color_red
        }
        max_hair_color = max(hair_color_values.values())
        if max_hair_color==0:
            continue
        hair_color = [k for k, v in hair_color_values.items() if v == max_hair_color][0]
        

        hairtype_values = {
            'coily': hairtype_coily,
            'dreadlocks': hairtype_dreadlocks,
            'bald': hairtype_bald,
            'straight': hairtype_straight,
            'curly': hairtype_curly,
            'wavy': hairtype_wavy        }
        max_hairtype = max(hairtype_values.values())
        if max_hairtype==0:
            continue
        hairtype = [k for k, v in hairtype_values.items() if v == max_hairtype][0]
        data_entry = {
            "filename": row["filename"],
            "class1": row["class1"],
            "gender_presentation_masc": row["gender_presentation_masc"],
            "gender_presentation_fem": row["gender_presentation_fem"],
            "skin_tone": max_skin_no,
            "hair_color": hair_color,
            "hairtype": hairtype
        }
        
        processed_data.append(data_entry)

with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

print(f"Processing complete! Total records processed: {len(processed_data)}")
print(f"Results saved to: {output_file}")
