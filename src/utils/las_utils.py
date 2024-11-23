import numpy as np
import laspy
from numba import njit, prange

def load_las_file(file_path):
    """
    Загружает LAS-файл и возвращает координаты точек как numpy массив.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    points_local = points - las.header.offset
    return points_local 

@njit(parallel=True)
def extract_features_from_object_numba(points, center, size, bounds):
    """
    Извлекает статистические признаки из точек внутри ограничивающего бокса.
    
    Args:
        points (np.ndarray): Массив точек [N, 3]
        center (tuple): Центр бокса (x, y, z)
        size (tuple): Размеры бокса (x, y, z)
        bounds (tuple): Границы LAS файла
        
    Returns:
        np.ndarray: Массив признаков
    """
    center_x, center_y, center_z = center
    size_x, size_y, size_z = size
    min_x, max_x, min_y, max_y, min_z, max_z = bounds

    # Определение границ бокса с учетом границ LAS
    box_min_x = max(center_x - size_x / 2, min_x)
    box_max_x = min(center_x + size_x / 2, max_x)
    box_min_y = max(center_y - size_y / 2, min_y)
    box_max_y = min(center_y + size_y / 2, max_y)
    box_min_z = max(center_z - size_z / 2, min_z)
    box_max_z = min(center_z + size_z / 2, max_z)

    # Фильтрация точек внутри бокса
    mask = (
        (points[:, 0] >= box_min_x) & (points[:, 0] <= box_max_x) &
        (points[:, 1] >= box_min_y) & (points[:, 1] <= box_max_y) &
        (points[:, 2] >= box_min_z) & (points[:, 2] <= box_max_z)
    )
    filtered_points = points[mask]

    # Инициализация признаков
    features = np.zeros(10)  # Расширенный набор признаков
    
    if filtered_points.shape[0] > 0:
        # Базовые статистические признаки
        features[0] = np.mean(filtered_points[:, 2])  # mean_z
        features[1] = np.max(filtered_points[:, 2])   # max_z
        features[2] = np.min(filtered_points[:, 2])   # min_z
        features[3] = np.std(filtered_points[:, 2])   # std_z
        features[4] = size_x                          # size_x
        features[5] = size_y                          # size_y
        features[6] = size_z                          # size_z
        
        # Дополнительные признаки
        features[7] = filtered_points.shape[0]         # количество точек
        features[8] = np.median(filtered_points[:, 2]) # median_z
        features[9] = np.var(filtered_points[:, 2])    # variance_z

    return features 

@njit(parallel=True)
def cluster_heights_numba(z_coords, min_cluster_size=5, max_gap=10.0, lep_gap_multiplier=3.0):
    """
    Кластеризация точек по высоте для обнаружения отдельных объектов с использованием Numba,
    включая обработку раздельных кластеров для ЛЭП, которые находятся далеко друг от друга.
    
    Args:
        z_coords (np.ndarray): Координаты Z точек
        min_cluster_size (int): Минимальный размер кластера
        max_gap (float): Максимальный разрыв между кластерами
        lep_gap_multiplier (float): Множитель для увеличения разрыва между кластерами ЛЭП
        
    Returns:
        np.ndarray: Массив меток кластеров
    """
    n_points = z_coords.size
    if n_points < min_cluster_size:
        return np.empty(0, dtype=np.int32)
    
    # Сортировка координат Z и индексов
    sorted_indices = np.argsort(z_coords)
    z_sorted = z_coords[sorted_indices]
    
    # Инициализация переменных для кластеризации
    labels = np.full(n_points, -1, dtype=np.int32)
    current_label = 0
    start_idx = 0
    
    # Определение динамического увеличения разрыва для кластеров ЛЭП
    dynamic_max_gap = max_gap
    is_lep_cluster = False  # Флаг для идентификации кластеров ЛЭП (примерно по значению высоты)
    
    for i in range(1, n_points):
        gap = z_sorted[i] - z_sorted[i - 1]
        
        # Условное увеличение разрыва для ЛЭП, если определено (например, по высоте)
        if z_sorted[i] > 50:  # Предполагается, что объекты ЛЭП выше определенного значения
            is_lep_cluster = True
            dynamic_max_gap = max_gap * lep_gap_multiplier
        else:
            is_lep_cluster = False
            dynamic_max_gap = max_gap
        
        if gap > dynamic_max_gap:
            cluster_size = i - start_idx
            if cluster_size >= min_cluster_size:
                labels[start_idx:i] = current_label
                current_label += 1
            start_idx = i
    
    # Обработка последнего кластера
    if n_points - start_idx >= min_cluster_size:
        labels[start_idx:n_points] = current_label
    
    # Восстановление меток в исходном порядке
    original_labels = np.full(n_points, -1, dtype=np.int32)
    for i in prange(n_points):
        original_labels[sorted_indices[i]] = labels[i]
    
    return original_labels

@njit
def estimate_yaw_numba(points):
    """
    Оценка ориентации объекта по точкам с использованием Numba.
    
    Args:
        points (np.ndarray): Точки объекта (N, 3)
        
    Returns:
        float: Угол ориентации в радианах
    """
    num_points, dim = points.shape
    if num_points < 3:
        return 0.0
    
    # Центрирование точек
    mean_x = 0.0
    mean_y = 0.0
    for i in range(num_points):
        mean_x += points[i, 0]
        mean_y += points[i, 1]
    mean_x /= num_points
    mean_y /= num_points
    
    centered_x = np.empty(num_points, dtype=np.float64)
    centered_y = np.empty(num_points, dtype=np.float64)
    for i in range(num_points):
        centered_x[i] = points[i, 0] - mean_x
        centered_y[i] = points[i, 1] - mean_y
    
    # Вычисление ковариационной матрицы
    cov_xx = 0.0
    cov_xy = 0.0
    cov_yy = 0.0
    for i in range(num_points):
        cov_xx += centered_x[i] * centered_x[i]
        cov_xy += centered_x[i] * centered_y[i]
        cov_yy += centered_y[i] * centered_y[i]
    cov_xx /= (num_points - 1)
    cov_xy /= (num_points - 1)
    cov_yy /= (num_points - 1)
    
    # Вычисление собственных значений
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    temp = (trace / 2.0) ** 2 - det
    if temp < 0:
        temp = 0.0  # Избегаем отрицательного значения под корнем
    sqrt_val = np.sqrt(temp)
    eigenvalue1 = trace / 2.0 + sqrt_val
    eigenvalue2 = trace / 2.0 - sqrt_val
    
    # Выбор главного направления
    if eigenvalue1 >= eigenvalue2:
        principal_x = cov_xx - eigenvalue2
        principal_y = cov_xy
    else:
        principal_x = cov_xx - eigenvalue1
        principal_y = cov_xy
    
    # Нормализация вектора
    norm = np.sqrt(principal_x ** 2 + principal_y ** 2)
    if norm == 0.0:
        return 0.0
    principal_x /= norm
    principal_y /= norm
    
    # Вычисление угла
    yaw = np.arctan2(principal_y, principal_x)
    return yaw

@njit
def compute_iou_numba(pred1, pred2):
    """
    Вычисление Intersection over Union (IoU) для двух предсказаний.
    
    Args:
        pred1 (float[:]): Массив предсказания 1 [center_x, center_y, center_z, size_x, size_y, size_z]
        pred2 (float[:]): Массив предсказания 2 [center_x, center_y, center_z, size_x, size_y, size_z]
    
    Returns:
        float: Значение IoU
    """
    # Извлечение координат и размеров
    min_x1 = pred1[0] - pred1[3] / 2
    max_x1 = pred1[0] + pred1[3] / 2
    min_y1 = pred1[1] - pred1[4] / 2
    max_y1 = pred1[1] + pred1[4] / 2
    min_z1 = pred1[2] - pred1[5] / 2
    max_z1 = pred1[2] + pred1[5] / 2
    
    min_x2 = pred2[0] - pred2[3] / 2
    max_x2 = pred2[0] + pred2[3] / 2
    min_y2 = pred2[1] - pred2[4] / 2
    max_y2 = pred2[1] + pred2[4] / 2
    min_z2 = pred2[2] - pred2[5] / 2
    max_z2 = pred2[2] + pred2[5] / 2
    
    inter_min_x = max(min_x1, min_x2)
    inter_max_x = min(max_x1, max_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_y = min(max_y1, max_y2)
    inter_min_z = max(min_z1, min_z2)
    inter_max_z = min(max_z1, max_z2)
    
    inter_volume = max(0.0, inter_max_x - inter_min_x) * \
                   max(0.0, inter_max_y - inter_min_y) * \
                   max(0.0, inter_max_z - inter_min_z)
    
    volume1 = pred1[3] * pred1[4] * pred1[5]
    volume2 = pred2[3] * pred2[4] * pred2[5]
    
    union_volume = volume1 + volume2 - inter_volume
    if union_volume == 0.0:
        return 0.0
    return inter_volume / union_volume 

@njit(parallel=True)
def non_maximum_suppression_numba(predictions, iou_threshold=0.5):
    num_preds = predictions.shape[0]
    keep = np.zeros(num_preds, dtype=np.int32)
    keep_count = 0
    is_kept = np.ones(num_preds, dtype=np.int8)

    for i in prange(num_preds):
        if is_kept[i]:
            keep[keep_count] = i
            keep_count += 1
            for j in range(i + 1, num_preds):
                if is_kept[j]:
                    iou = compute_iou_numba(predictions[i, :6], predictions[j, :6])
                    if iou > iou_threshold:
                        is_kept[j] = 0
    return keep[:keep_count]
