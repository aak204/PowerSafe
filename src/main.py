import argparse
import os
from engine.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="PowerSafe Inference Script")
    parser.add_argument('--test_dir', type=str, required=True, help='Путь к директории с тестовыми LAS файлами')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Путь для сохранения результатов')
    parser.add_argument('--model_path', type=str, default='src/model/model.pkl', help='Путь к файлу модели')
    parser.add_argument('--scaler_path', type=str, default='src/model/scaler.pkl', help='Путь к файлу скейлера')
    
    args = parser.parse_args()
    
    engine = InferenceEngine(model_path=args.model_path, scaler_path=args.scaler_path)
    engine.process_test_directory(test_dir=args.test_dir, output_path=args.output_path)

if __name__ == "__main__":
    main()
