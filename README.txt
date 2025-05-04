## ConvNeXt-Base Classifier (Linear Probing)

This project uses a pretrained ConvNeXt-Base model from torchvision for binary image classification (`real` vs `fake`) using linear probing.

## 🧠 Methodology

- Base model: `ConvNeXt-Base` pretrained on ImageNet
- Encoder kept **frozen** during training
- Trained a `Linear(1024, 2)` classification head on top
- Efficient and fast — ideal for small datasets or fast prototyping

## 📁 Dataset Structure

data/
├── train/
│ ├── real/
│ └── fake/
├── val/
│ ├── real/
│ └── fake/
└── test/ 
  ├── real/
  └── fake/


```bash command
    python train_convo.py


-Batch size: 32
-Epochs: 3–5
-Optimizer: Adam
-Learning rate: 1e-3
-Backbone frozen
-Trainable head only: Linear(1024, 2)
-Progress shown with tqdm

## 📊 Results

Set	                Accuracy	  Misclassified
Validation	            --              --
data/field	            --              --
Confusion Matrix (Field Set)
        [[700   0]
        [  0 774]]
## 🧪 Evaluation

```bash command
        python evalute_convo.py


Outputs accuracy, precision, recall, and F1
Saves misclassified image paths to misclassified_field.txt (if any)




---




## Fine-Tuned ConvNeXt-Base Classifier

This project fine-tunes a ConvNeXt-Base model from torchvision for binary classification: `real` vs `fake`.

## 📦 Model

- Base: `ConvNeXt-Base` (ImageNet pretrained)
- Output embedding: 1024-dim
- Fine-tuned: Full encoder + classification head (`Linear(1024, 2)`)
- Framework: PyTorch

## 📁 Dataset Structure

data/
├── train/
│ ├── real/
│ └── fake/
├── val/
│ ├── real/
│ └── fake/
└── field/
├── real/
└── fake/


```bash command
        python train_convo_finetune.py



-Optimizer: Adam
-Learning rate: 1e-4 (full model)
-Epochs: 3
-Progress tracked with tqdm


## 📊 Results

Set	                    Accuracy	            Misclassified
Validation	            99.96%	                    -
data/field             	100.00%	                    0
Confusion Matrix (Field Set)
            [[700   0]
            [  0 774]]
## 🧪 Evaluation
 bash command
            python evalute_convnext.py
Outputs:

Accuracy, precision, recall, f1-score
Misclassified image paths saved to misclassified_field.txt (if any)
