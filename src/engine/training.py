import argparse
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.las_utils import load_las_file, extract_features_from_object_numba
from engine.inference import PowerLineDetector

def prepare_training_data(detector, annotations_df, las_files_dir):
    return detector.prepare_training_data(annotations_df, las_files_dir)

def train_model(feature_df, model_path, scaler_path):
    detector = PowerLineDetector(use_neural_network=False)
    detector.scaler.fit(feature_df[detector.feature_columns].values)
    X_scaled = detector.scaler.transform(feature_df[detector.feature_columns].values)
    detector.model.fit(X_scaled, feature_df['class'].values)
  
    print(f"Точность на обучающих данных: {detector.model.score(X_scaled, feature_df['class'].values):.4f}")

def main():
    parser = argparse.ArgumentParser(description="PowerSafe Training Script")
    parser.add_argument('--train_dir', type=str, required=True, help='Путь к директории с обучающими LAS файлами')
    parser.add_argument('--annotations', type=str, required=True, help='Путь к файлу аннотаций train.csv')
    parser.add_argument('--model_path', type=str, default='src/model/model.pkl', help='Путь для сохранения модели')
    parser.add_argument('--scaler_path', type=str, default='src/model/scaler.pkl', help='Путь для сохранения скейлера')
    
    args = parser.parse_args()
    
    annotations_df = pd.read_csv(args.annotations)
    detector = PowerLineDetector(use_neural_network=False)
    
    print("Подготовка обучающих данных...")
    feature_df = prepare_training_data(detector, annotations_df, args.train_dir)
    
    print("Обучение модели...")
    train_model(feature_df, args.model_path, args.scaler_path)

if __name__ == "__main__":
    main()
