# Few-Shot-and-Zero-Shot-Learning
Few-shot learning aims to train models with very few labeled examples, while zero-shot learning enables models to generalize to unseen classes based on learned semantic relationships. These approaches are becoming increasingly important in real-world tasks where labeled data is hard to obtain.

# Few‑Shot & Zero‑Shot Learning on CIFAR‑10

This repository demonstrates both few‑shot and zero‑shot classification on CIFAR‑10:

* **Few‑Shot**: Train a linear classifier on top of a frozen ResNet‑18 backbone using only 1, 5, or 10 labeled examples per class.
* **Zero‑Shot**: Use OpenAI’s CLIP (ViT‑B/32) to classify CIFAR‑10 images without any training on the task.


## 🚀 Features

* **Data Loading**

  * CIFAR‑10 train/test split with standard normalization and resizing.
* **Few‑Shot Pipeline**

  * Sample `K` shots per class from the training set.
  * Extract features via pretrained ResNet‑18 (backbone).
  * Train a lightweight linear head.
  * Evaluate on the full CIFAR‑10 test set.
* **Zero‑Shot Pipeline**

  * Encode class names with CLIP’s text encoder.
  * Encode test images with CLIP’s image encoder.
  * Compute cosine similarities for classification.
* **Visualization**

  * Bar chart comparing accuracies of 1‑shot, 5‑shot, 10‑shot, and zero‑shot.



## 📋 Requirements

* Python 3.7+
* PyTorch 1.10+ & TorchVision 0.11+
* Transformers 4.10+
* matplotlib
* tqdm

Install with:

```bash
pip install torch torchvision transformers matplotlib tqdm
```



## 📁 File

* `few_zero_learning.py`
  Full end‑to‑end script. Contains data loading, few‑shot & zero‑shot routines, evaluation, and plotting.


## ▶️ Usage

1. **Download dependencies**:

   ```bash
   pip install torch torchvision transformers matplotlib tqdm
   ```

2. **Run the script**:

   ```bash
   python few_zero_learning.py
   ```

   * Downloads CIFAR‑10 automatically.
   * Prints 1‑shot, 5‑shot, 10‑shot, and zero‑shot accuracies.
   * Saves a bar chart as `outputs/few_zero_shot_comparison.png`.



## 📊 Example Output

```
=== 1‑shot experiment ===
1‑shot test accuracy: 15.20%
=== 5‑shot experiment ===
5‑shot test accuracy: 35.40%
=== 10‑shot experiment ===
10‑shot test accuracy: 50.85%

=== zero‑shot via CLIP ===
zero‑shot test accuracy: 29.73%
```

![Comparison Plot](outputs/few_zero_shot_comparison.png)


## 🔍 Notes

* Few‑shot models train **only** the final linear layer; backbone is frozen.
* Zero‑shot uses standard CLIP prompts:

  > “a photo of a {class\_name}”
* Results may vary depending on hardware and random seed.


## 🧪 Reproducibility

* Random seed is fixed (`42`) for both PyTorch and NumPy.
* Batch size, learning rate, and epochs are configurable at the top of the script.
