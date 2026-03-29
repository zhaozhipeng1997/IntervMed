# Data Preparation

Before running the code, you need to prepare the evaluation datasets.

- **FACET dataset**: Download from https://ai.meta.com/datasets/facet-downloads/
- **MS COCO dataset**:
  - Images: http://images.cocodataset.org/zips/test2014.zip
  - Labels: https://princetonvisualai.github.io/imagecaptioning-bias/

Both evaluation datasets require filling out a simple application form. We provide a well-organized dataset folder structure as follows:

```
FACET_Dataset/
├── annotations/
│   ├── annotations.csv
│   ├── coco_boxes.json
│   ├── coco_masks.json
│   └── README.txt
├── images/
│   ├── sa_385.jpg
│   ├── sa_582.jpg
│   ├── ...
```

```
COCO Racial Biases in Image Captioning/
├── images_val2014.csv
├── instances_2014all.csv
├── README.pdf

val2014/
├── COCO_val2014_000000000042.jpg
├── COCO_val2014_000000000073.jpg
├── ...
```

# Experiment Configuration

After downloading the evaluation datasets, configure them in the corresponding configuration files:

- For **InstructBLIP 7B** and **InstructBLIP 13B** experiments:
  - FACET: Line 5 in `instructblip/utils/generation_config.yaml`
  - MS COCO: Lines 9-10 in `instructblip/utils/generation_config.yaml`

- For **LLaVA-1.5 7B/13B** and **LLaVa-NeXT 7B/13B** experiments:
  - FACET: Line 5 in `llava/utils/generation_config.yaml`
  - MS COCO: Lines 11-12 in `llava/utils/generation_config.yaml`

# Environment Setup

We provide environment installation files. Run the following command:

```bash
pip install -r requirments.txt && cd transformers-4.45.0 && pip install -e .
```

# Running Experiments

We provide convenient one-click scripts:

## Evaluating All Layers

```bash
cd llava && ./run_all.sh
```

```bash
cd instructblip && ./run_all.sh
```

## Intervention Strength Experiments

```bash
cd llava && ./run_Intervention_exp.sh
```

```bash
cd instructblip && ./run_Intervention_exp.sh
```