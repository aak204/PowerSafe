import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from numba import njit, prange
import laspy
import os
from tqdm import tqdm
import gc
from scipy.spatial import KDTree
import pickle
from utils.las_utils import (
    load_las_file,
    extract_features_from_object_numba,
    cluster_heights_numba,
    estimate_yaw_numba,
    compute_iou_numba,
    non_maximum_suppression_numba
)

class PowerLineDetector:
    """
    Класс для обнаружения ЛЭП и растительности в облаках точек.
    """
    
    def __init__(self, use_neural_network=False):
        """
        Инициализация детектора.
        
        Args:
            use_neural_network (bool): Использовать нейронную сеть вместо случайного леса
        """
        if use_neural_network:
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.feature_columns = [
            'mean_z', 'max_z', 'min_z', 'std_z', 
            'size_x', 'size_y', 'size_z', 'n_points',
            'median_z', 'var_z', 'yaw'
        ]
    
    def sliding_window_detection(self, points, bounds):
        """
        Улучшенная генерация кандидатов для множественных объектов.
        
        Args:
            points (np.ndarray): Облако точек
            bounds (tuple): Границы области
            
        Returns:
            list: Список кандидатов
        """
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        
        # Конфигурации окон для разных классов объектов
        window_configs = np.array([
            [10, 10, 70, 5, 15, 'LEP_metal'],   # Высокие и узкие окна для ЛЭП металлических
            [8, 8, 60, 4, 12, 'LEP_prom'],     # Узкие окна для ЛЭП промышленных
            [20, 20, 15, 10, 30, 'vegetation'] # Низкие и более широкие окна для кустарников
        ], dtype=object)

        # Создаем KD-дерево для эффективного поиска точек
        tree = KDTree(points)
        candidates = []
        
        for config in window_configs:
            size_x, size_y, size_z, stride, min_points, class_type = config
            x_steps = np.arange(min_x, max_x, stride)
            y_steps = np.arange(min_y, max_y, stride)
            
            for x in x_steps:
                for y in y_steps:
                    local_points_idx = tree.query_ball_point(
                        [x, y, (min_z + max_z)/2], 
                        max(size_x, size_y)/2
                    )
                    
                    if len(local_points_idx) < min_points:
                        continue
                    
                    local_points = points[local_points_idx]
                    z_coords = local_points[:, 2]
                    
                    # Кластеризация высот с использованием Numba
                    cluster_labels = cluster_heights_numba(z_coords)
                    
                    unique_clusters = np.unique(cluster_labels)
                    for cluster_id in unique_clusters:
                        if cluster_id == -1:
                            continue
                        cluster_points = local_points[cluster_labels == cluster_id]
                        center_z = cluster_points[:, 2].mean()
                        center = (x, y, center_z)
                        
                        # Извлечение признаков
                        features = extract_features_from_object_numba(
                            cluster_points, center, (size_x, size_y, size_z), bounds
                        )
                        
                        if not np.any(np.isnan(features)) and features[7] >= min_points:
                            # Оценка yaw с использованием Numba
                            yaw = estimate_yaw_numba(cluster_points)
                            
                            candidate = {
                                'center_x': x,
                                'center_y': y,
                                'center_z': center_z,
                                'size_x': size_x,
                                'size_y': size_y,
                                'size_z': size_z,
                                'yaw': yaw,
                                'features': features,
                                'suggested_class': class_type,
                                'n_points': features[7]
                            }
                            candidates.append(candidate)
        
        return candidates

    def predict_test_file(self, las_path):
        """
        Обработка тестового файла с улучшенной обработкой множественных объектов.

        Args:
            las_path (str): Путь к тестовому LAS файлу

        Returns:
            list: Список предсказаний
        """
        # Проверка, был ли обучен скейлер
        if not hasattr(self.scaler, 'mean_'):
            raise AttributeError("StandardScaler не обучен. Вызовите метод 'fit' перед предсказанием.")

        points = load_las_file(las_path)

        bounds = (
            points[:, 0].min(), points[:, 0].max(),
            points[:, 1].min(), points[:, 1].max(),
            points[:, 2].min(), points[:, 2].max()
        )
        candidates = self.sliding_window_detection(points, bounds)
        predictions = []
        current_id = 1

        # Группировка кандидатов по классу
        class_candidates = {}
        for candidate in candidates:
            class_type = candidate['suggested_class']
            if class_type not in class_candidates:
                class_candidates[class_type] = []
            class_candidates[class_type].append(candidate)

        # Словарь для хранения предсказаний после классификации
        class_predictions = {}

        for class_type, class_candidates_list in class_candidates.items():
            # Сортировка кандидатов по количеству точек
            class_candidates_list.sort(key=lambda x: x['n_points'], reverse=True)

            # Подготовка данных для масштабирования и предсказания
            features_list = []
            for candidate in class_candidates_list:
                features = np.append(candidate['features'], candidate['yaw'])
                features_list.append(features)

            if not features_list:
                continue

            features_array = np.vstack(features_list)

            # Применение уже обученного скейлера
            features_scaled = self.scaler.transform(features_array)

            # Предсказание вероятностей
            pred_probas = self.model.predict_proba(features_scaled)
            pred_classes = self.model.classes_[np.argmax(pred_probas, axis=1)]
            confidences = np.max(pred_probas, axis=1)

            # Применение порогов уверенности
            confidence_thresholds = {
                'LEP_metal': 0.35,
                'LEP_prom': 0.35,
                'vegetation': 0.3
            }

            filtered_preds = []
            for i, candidate in enumerate(class_candidates_list):
                confidence = confidences[i]
                pred_class = pred_classes[i]
                threshold = confidence_thresholds.get(class_type, 0.3)

                if confidence > threshold:
                    filtered_preds.append({
                        'center_x': candidate['center_x'],
                        'center_y': candidate['center_y'],
                        'center_z': candidate['center_z'],
                        'size_x': candidate['size_x'],
                        'size_y': candidate['size_y'],
                        'size_z': candidate['size_z'],
                        'yaw': candidate['yaw'],
                        'score': confidence,
                        'class': pred_class,
                        'id': current_id,
                        'file_name': os.path.basename(las_path)
                    })
                    current_id += 1

            # Сохранение предсказаний отдельно
            if class_type not in class_predictions:
                class_predictions[class_type] = []
            class_predictions[class_type].extend(filtered_preds)

        all_filtered_predictions = []
        for class_type in ['LEP_metal', 'LEP_prom', 'vegetation']:
            class_preds = class_predictions.get(class_type, [])
            all_filtered_predictions.extend(class_preds)

        # Сортировка предсказаний по ID
        all_filtered_predictions.sort(key=lambda x: x['id'])

        del points
        gc.collect()

        return all_filtered_predictions

    def prepare_training_data(self, annotations_df, las_files_dir):
        """
        Подготовка обучающих данных из аннотаций.
        
        Args:
            annotations_df (pd.DataFrame): DataFrame с аннотациями
            las_files_dir (str): Путь к директории с LAS файлами
            
        Returns:
            pd.DataFrame: DataFrame с признаками и метками
        """
        data = []
        unique_files = annotations_df['file_name'].unique()
        
        for file in tqdm(unique_files, desc="Подготовка обучающих данных"):
            file_path = os.path.join(las_files_dir, file)
            
            if not os.path.exists(file_path):
                print(f"Файл {file_path} не найден")
                continue
                
            try:
                points = load_las_file(file_path)
                bounds = (
                    points[:, 0].min(), points[:, 0].max(),
                    points[:, 1].min(), points[:, 1].max(),
                    points[:, 2].min(), points[:, 2].max()
                )
                
                # Фильтрация аннотаций для текущего файла
                file_annotations = annotations_df[annotations_df['file_name'] == file].copy()
                file_annotations.rename(columns={'class': 'target'}, inplace=True)
                
                for row in file_annotations.itertuples():
                    center = (row.center_x, row.center_y, row.center_z)
                    size = (row.size_x, row.size_y, row.size_z)
                    
                    features = extract_features_from_object_numba(points, center, size, bounds)
                    
                    if not np.any(np.isnan(features)):
                        feature_dict = {
                            'mean_z': features[0],
                            'max_z': features[1],
                            'min_z': features[2],
                            'std_z': features[3],
                            'size_x': features[4],
                            'size_y': features[5],
                            'size_z': features[6],
                            'n_points': features[7],
                            'median_z': features[8],
                            'var_z': features[9],
                            'yaw': row.yaw,
                            'class': row.target,
                            'file_name': file
                        }
                        data.append(feature_dict)
            
            except Exception as e:
                print(f"Ошибка при обработке файла {file}: {e}")
                continue
                
            finally:
                del points
                gc.collect()
        
        return pd.DataFrame(data)

class InferenceEngine:
    """
    Класс для выполнения инференса с загруженной моделью и скейлером.
    """
    def __init__(self, model_path, scaler_path):
        # Загрузка модели
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Загрузка скейлера
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.detector = PowerLineDetector(use_neural_network=False)
        self.detector.model = self.model
        self.detector.scaler = self.scaler
    
    def process_test_directory(self, test_dir, output_path='submission.csv'):
        """
        Обработка всей тестовой директории и создание файла с предсказаниями.
        
        Args:
            test_dir (str): Путь к директории с тестовыми файлами
            output_path (str): Путь для сохранения результатов
            
        Returns:
            pd.DataFrame: DataFrame с предсказаниями
        """
        all_predictions = []
        current_id = 1
        
        for file_name in tqdm(os.listdir(test_dir), desc="Обработка тестовых файлов"):
            if file_name.endswith('.las'):
                file_path = os.path.join(test_dir, file_name)
                try:
                    predictions = self.detector.predict_test_file(file_path)
                    all_predictions.extend(predictions)
                except Exception as e:
                    print(f"Ошибка при обработке {file_name}: {e}")
                gc.collect()
        
        # Создание DataFrame для отправки
        submission_df = pd.DataFrame(all_predictions)
        
        # Проверка наличия всех необходимых столбцов
        required_columns = [
            'id', 'file_name', 'center_x', 'center_y', 'center_z',
            'size_x', 'size_y', 'size_z', 'yaw', 'class', 'score'
        ]
        
        for col in required_columns:
            if col not in submission_df.columns:
                print(f"Внимание: отсутствует столбец {col}")
        
        # Сохранение результатов
        submission_df.to_csv(output_path, index=False)
        print(f"Сохранены предсказания в {output_path}")
        return submission_df
