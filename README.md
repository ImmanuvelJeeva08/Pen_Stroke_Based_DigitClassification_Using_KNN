# 🧠 Pen Stroke-Based Digit Classification (KNN)

## 🚀 Overview

This project implements a **Machine Learning classification system from scratch in Java** to recognize handwritten digits using **pen stroke coordinate data**.

Unlike image-based approaches, this project works on **(X, Y) coordinate sequences** that capture how a digit is written, rather than how it looks as pixels.

---

## 📊 Dataset Explanation

### 📌 Dataset Used

* UCI Pendigits Dataset
* Represents handwritten digits (0–9)

---

## 🔍 Data Representation

Each row in the dataset contains:

```
x1 y1 x2 y2 x3 y3 ... x8 y8 label
```

### ✔️ Breakdown:

* **16 features** → 8 points of (X, Y) coordinates
* **1 label** → Digit (0–9)

---

### 🧾 Example Row

```
10 100 12 90 14 80 16 70 18 60 20 50 22 40 24 30 1
```

### 👉 Meaning:

* (10,100), (12,90), ..., (24,30) → Pen movement
* **1** → Digit label

---

## 🖊️ How the Data Works (Visualization)

### Step 1: Raw Coordinates

```
(10,100)
(12,90)
(14,80)
(16,70)
...
```

### Step 2: Plot Points on Grid

```
   y
100 |   *
 90 |   *
 80 |   *
 70 |   *
 60 |   *
     ---------
       x
```

### Step 3: Connect Points

```
   *
   *
   *
   *
   *
```

👉 This forms digit **“1”**

---

## 🧠 Approach

### 1️⃣ K-Nearest Neighbors (KNN)

* Computes distance between test and training points
* Selects **K nearest neighbors**
* Uses **majority voting**

### Distance Formula:

* Uses **squared Euclidean distance** (optimized)

---

## ⚙️ Features

* ✅ Built ML algorithms **from scratch (no libraries)**
* ✅ Handles **data preprocessing**
* ✅ Implements **distance-based learning**
* ✅ Supports configurable **K values**
* ✅ Outputs prediction results with accuracy

---

## 📈 Output Format

Example result file:

```
Index  Predicted  Actual  Correct
1      1          1       1
2      3          2       0
```

---

## 🎯 Key Learnings

* Understanding **distance-based vs probability-based models**
* Importance of **feature representation**
* Implementing ML logic **without frameworks**
* Handling real-world dataset formats

---

## 💡 Key Insight

This project demonstrates that:

> Handwritten digit recognition can be performed using **pen stroke motion (coordinates)** instead of images, making it lightweight and efficient.

---

## 🔗 Future Improvements

* Visualize digits as images (PNG output)
* Compare KNN vs Bayesian accuracy
* Add GUI visualization
* Optimize using KD-Trees

---
