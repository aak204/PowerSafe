# PowerSafe: ИИ для Защиты Охранных Зон ЛЭП

**Ноябрь 2024**

## Описание

**PowerSafe** — это система искусственного интеллекта, предназначенная для защиты охранных зон линий электропередач (ЛЭП). Она обрабатывает облака точек из LAS-файлов, выполняет предсказания объектов и генерирует визуальные результаты вместе с CSV-файлом предсказаний.

## Структура Репозитория

```
PowerSafe/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── engine/
    │   ├── __init__.py
    │   ├── training.py
    │   └── inference.py
    ├── model/
    │   ├── model.pkl
    │   └── scaler.pkl
    └── utils/
        ├── __init__.py
        └── las_utils.py
```

## Установка

1. **Клонируйте репозиторий:**

    ```bash
    git clone https://github.com/ваш-репозиторий/PowerSafe.git
    cd PowerSafe
    ```

2. **Создайте виртуальное окружение и активируйте его:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Для Windows: venv\Scripts\activate
    ```

3. **Установите зависимости:**

    ```bash
    pip install -r requirements.txt
    ```

## Обучение Модели

1. **Подготовьте данные:**

    Убедитесь, что у вас есть директория с обучающими LAS-файлами и файл аннотаций `train.csv`.

2. **Запустите скрипт обучения:**

    ```bash
    python src/engine/training.py --train_dir путь/к/train_dir --annotations путь/к/train.csv --model_path src/model/model.pkl --scaler_path src/model/scaler.pkl
    ```

    Это создаст файлы `model.pkl` и `scaler.pkl` в директории `src/model/`.

## Инференс

1. **Запустите основной скрипт для инференса:**

    ```bash
    python src/main.py --test_dir путь/к/test_dir --output_path путь/к/submission.csv --model_path src/model/model.pkl --scaler_path src/model/scaler.pkl
    ```

    Это создаст файл `submission.csv` с предсказаниями.

## Веса Модели

Веса модели можно скачать по [ссылке](https://example.com/model-weights).

## Постобработка

Для применения постобработки, такой как подавление немаксимумов (Non-Maximum Suppression), убедитесь, что соответствующие функции реализованы и интегрированы в процесс инференса. В данном проекте постобработка уже включена в скрипт инференса.

## Пример Использования

### Обучение Модели

```bash
python src/engine/training.py --train_dir data/train/ --annotations data/train.csv --model_path src/model/model.pkl --scaler_path src/model/scaler.pkl
```

### Инференс на Тестовых Данных

```bash
python src/main.py --test_dir data/test/ --output_path results/submission.csv --model_path src/model/model.pkl --scaler_path src/model/scaler.pkl
```

## Зависимости

Все необходимые библиотеки перечислены в файле `requirements.txt`. Основные из них:

- `numpy`
- `pandas`
- `scikit-learn`
- `numba`
- `laspy`
- `tqdm`
- `scipy`

## Лицензия

Этот проект лицензирован под MIT License. Подробнее см. [LICENSE](LICENSE).
