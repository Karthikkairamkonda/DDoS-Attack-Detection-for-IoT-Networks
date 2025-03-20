import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Global variables
dataset = None
X_train, X_test, y_train, y_test = None, None, None, None
classifier = None
labels = []

# Initialize GUI
root = tk.Tk()
root.title("DDoS Attack Detection in IoT Networks")
root.geometry("600x400")

# Function to load dataset
def load_dataset():
    global dataset, labels
    folder_path = filedialog.askdirectory(title="Select Dataset Folder")
    if not folder_path:
        return
    
    try:
        files = ["DrDOS_DNS.csv", "DrDOS_LDAP.csv", "Syn.csv"]  # Add all dataset files
        data_frames = [pd.read_csv(os.path.join(folder_path, f)) for f in files if os.path.exists(os.path.join(folder_path, f))]
        if not data_frames:
            messagebox.showerror("Error", "No valid dataset files found!")
            return
        dataset = pd.concat(data_frames, ignore_index=True)
        labels = dataset['Label'].unique().tolist()
        messagebox.showinfo("Success", "Dataset Loaded Successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

# Function to preprocess dataset
def preprocess_data():
    global dataset, X_train, X_test, y_train, y_test
    if dataset is None:
        messagebox.showerror("Error", "Please load the dataset first")
        return
    
    try:
        dataset.fillna(0, inplace=True)
        
        # Convert categorical columns to numeric
        for column in dataset.columns:
            if dataset[column].dtype == 'object':  # If the column contains strings
                le = LabelEncoder()
                dataset[column] = le.fit_transform(dataset[column].astype(str))
        
        X = dataset.drop(columns=['Label'])
        Y = dataset['Label']
        X = normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        messagebox.showinfo("Success", "Data Preprocessing Completed")
    except Exception as e:
        messagebox.showerror("Error", f"Data preprocessing failed: {str(e)}")

# Function to train a model
def train_model(model_name):
    global classifier
    if X_train is None:
        messagebox.showerror("Error", "Please preprocess the data first")
        return
    
    models = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=3)
    }
    
    try:
        classifier = models.get(model_name)
        if classifier is None:
            messagebox.showerror("Error", "Invalid model selection")
            return
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        messagebox.showinfo("Training Complete", f"{model_name} Model Accuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Model training failed: {str(e)}")

# Function to display model performance
def show_performance():
    if classifier is None:
        messagebox.showerror("Error", "Train a model first")
        return
    
    predictions = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# GUI Elements
tk.Button(root, text="Load Dataset", command=load_dataset).pack(pady=10)
tk.Button(root, text="Preprocess Data", command=preprocess_data).pack(pady=10)
tk.Button(root, text="Train Naive Bayes", command=lambda: train_model("Naive Bayes")).pack(pady=5)
tk.Button(root, text="Train Random Forest", command=lambda: train_model("Random Forest")).pack(pady=5)
tk.Button(root, text="Show Performance", command=show_performance).pack(pady=10)

# Run GUI
root.mainloop()
