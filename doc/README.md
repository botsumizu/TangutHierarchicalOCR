
# TangutHierarchicalOCR
Botsumizu
## Introduction

**TangutHierarchicalOCR (THOCR)** is a hierarchical OCR framework designed for recognizing Tangut characters.
The system consists of two major components:

1. **Structure Classifier** – predicts the structural type of a Tangut character.
2. **Character Recognizer Package** – contains four specialized recognizers corresponding to each structure type.

The overall workflow is illustrated below:

```
Input -> StructureClassifier ──► E → EnclosedRecognizer     ─┐
                                 H → HorizontalRecognizer   ─┤→ Output
                                 V → VerticalRecognizer     ─┤
                                 S → SingleRecognizer       ─┘
```

---

## Structure Classification

Almost all Tangut characters can be categorized into four structural patterns:

```
Single (S)
Complex:
    Horizontal (H)
    Vertical (V)
    Enclosed (E)
```

Among them, **Single-type characters are relatively rare**, while the three complex types play a critical role in the overall dataset.

---

## Structure Classifier Pre-training

([PreTrain Script](preTrain/train_structure.py))

We use **ResNet-18** as the backbone for structure classification.
In the pre-training stage, the classifier is trained on a small dataset (20 training samples), yet still achieves **~90% accuracy** on a validation set composed entirely of unseen samples.

This result demonstrates the feasibility and effectiveness of our hierarchical approach.

![PreTrainDataGraph](preTrain/training_history.png)

---

## Structure Classifier Formal Training

([Structure Train Script](formalTrain/structureTrain/structureTrain.py))

After validating the feasibility, we expanded the dataset for formal training.
Because the Single-type is underrepresented, it contains only **20 training samples**, whereas the other three types have **50 samples each**.
The validation set includes **5 Single-type samples** and **20 samples for each other type**.

After **30 epochs of training**, the classifier reached a **best accuracy of ~98%**.
Even with the smallest amount of training data, the **Single-type** still achieved **~80% accuracy**.

![StructureTrainDataGraph](formalTrain/structureTrain/training_history_balanced.png)

---

## Character Recognizer Training

([Data Augmentation Script](formalTrain/recognizeTrain/datasetProcess/augment_dataset.py)
[Recognizer Train Script](formalTrain/recognizeTrain/train/recognizerTrain.py))

Before training each character recognizer, we performed **data augmentation** to increase sample diversity.
Raw images were reserved as the validation set to ensure unbiased evaluation.

We then trained four recognizers corresponding to the structure classifier outputs (E, H, S, V).

Training histories are shown below:

![ETrainDataGraph](formalTrain/recognizeTrain/train/training_recognizer_E_v2.png)
![HTrainDataGraph](formalTrain/recognizeTrain/train/training_recognizer_H_v2.png)
![STrainDataGraph](formalTrain/recognizeTrain/train/training_recognizer_S_v2.png)
![VTrainDataGraph](formalTrain/recognizeTrain/train/training_recognizer_V_v2.png)

---

## Integrated System

([THOCR Main File](integratedSystem/TangutHierarchicalOCR.py))

Finally, we integrated the structure classifier and all four recognizers into a complete OCR system — **THOCR**.

To evaluate the system, we created a test dataset of **50 samples**, and the integrated model achieved **96% recognition accuracy**.

![THOCRTestDataGraph](integratedSystem/test/thocr_performance_analysis.png)

---


