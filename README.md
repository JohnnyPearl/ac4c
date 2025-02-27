# Experimental Framework for Sequence Classification  

## 1. Experiment Design  

We aim to evaluate the performance of different models on sequence classification tasks using both **traditional machine learning** and **deep learning approaches**.  

### **a) Baseline Comparison: One-Hot Encoding + CNN vs. ML Models**  
- Train a simple **CNN** with **one-hot encoding** features.  
- Compare it against four traditional machine learning models:  
  - **K-Nearest Neighbors (KNN)**  
  - **Support Vector Machine (SVM)**  
  - **Random Forest (RF)**  
  - **Gradient Boosting (GB)**  
- Each ML model will be optimized using key hyperparameters:  
  - **KNN**: `n_neighbors = [3, 5, 7, 9, 11, 15]`  
  - **XGBoost**: `n_estimators = 100`, `learning_rate = [0.01, 0.05, 0.1]`  
  - **LightGBM**: `n_estimators = 100`, `learning_rate = [0.01, 0.05, 0.1]`  

### **b) Deep Learning Model Comparison**  
- Compare multiple deep learning architectures to evaluate their effectiveness.  
- Implement different **CNN**, **RNN**, or **Transformer-based** architectures.  

### **c) Performance & Robustness Evaluation**  
- **Performance:** Run multiple trials with different random seeds to analyze consistency.  
- **Robustness:** Split the dataset into different subsets and assess performance stability.  

### **d) Biological Analysis**  
- Conduct **logo analysis** to extract biologically meaningful patterns from sequences.  

---

## 2. Code Organization  

To ensure **modularity and reusability**, the experiment is divided into multiple **Python modules**:  

```bash
project/
│── models/               # Machine Learning & Deep Learning models
│   ├── cnn.py            # CNN model implementation
│   ├── knn.py            # KNN model implementation
│   ├── svm.py            # SVM model implementation
│   ├── random_forest.py  # Random Forest model
│   ├── gradient_boost.py # XGBoost & LightGBM models
│── utils/                # Helper functions
│   ├── utils.py 
│   ├── visualize.py       
│   ├── model.py
│   ├── ml_model.py
│── experiments/    
│   ├── exp1_feature_screening.ipynb 
│   ├── exp2_feature_screening.ipynb 
│   ├── exp3_feature_screening.ipynb 
│   ├── exp4_feature_screening.ipynb 
│   ├── run_baseline.py    
│   ├── run_deep_learning.py  
│   ├── robustness_test.py 
│   ├── biological_analysis.py 
│── config.py            
│── main.py             
│── requirements.txt       # Dependencies
│── README.md              # Documentation
```