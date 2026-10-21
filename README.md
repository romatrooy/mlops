# MLOps — Практическая работа № 3 «Инженерия машинного обучения»

Задача: Fashion-MNIST (вариант 3 — Zalando Store Fashion Image Dataset), три модели,
аугментация данных, сравнение экспериментов с аугментацией и без.
Инструменты: **Git/GitHub + DVC** (пайплайн, версионирование данных и экспериментов) и
**MLflow** (трекинг запусков).

## Структура проекта

```
data/raw/          # исходный Fashion-MNIST (DVC out, стадия download)
data/processed/    # трансформации + train/val сплит (стадия preprocess)
data/augmented/    # аугментированный датасет (стадия augment)
models/            # лучшие чекпоинты моделей
reports/           # metrics_*.json, comparison.csv/.md, figures/
src/data/download.py    # загрузка датасета
src/data/preprocess.py  # предобработка и сплит
src/data/augment.py     # аугментация + сохранение нового датасета
src/models/train.py     # обучение (3 модели x 2 набора данных) + MLflow
src/evaluate.py         # сравнение экспериментов и отчёт
dvc.yaml, params.yaml   # описание пайплайна и гиперпараметров
```

## Установка

```powershell
cd c:\Users\roman\MLOps
python -m pip install -r requirements.txt
python -m dvc init          # выполняется один раз (если .dvc ещё нет)
git add .dvc .dvcignore .gitignore
git commit -m "init dvc"
```

> В этом окружении консольные скрипты `dvc` и `mlflow` устанавливаются в
> `C:\Users\roman\AppData\Roaming\Python\Python313\Scripts` и не попадают в `PATH`,
> поэтому команды вызываются как `python -m dvc ...` и `python -m mlflow ...`.
> Если добавить этот каталог в `PATH`, можно писать просто `dvc` и `mlflow`.

## Быстрый запуск всего пайплайна

```powershell
python -m dvc repro          # download -> preprocess -> augment -> 6 x train -> evaluate
python -m dvc metrics show   # таблица метрик по всем экспериментам
python -m dvc dag            # граф пайплайна
```

`dvc repro` перезапускает только изменившиеся стадии, что и демонстрирует
воспроизводимость (пункт 3 требований к работе).

> Первый полный прогон создаёт `dvc.lock` (его нужно закоммитить) и генерирует
> 180 000 изображений в `data/augmented` — это самый долгий шаг.
> Учтите: `dvc repro --dry` на «холодном» состоянии (пока метрики ещё не созданы)
> завершается ошибкой `failed to reproduce 'evaluate'`; обычный `dvc repro` в этом
> же состоянии работает корректно.

## Ручной запуск по шагам

```powershell
python src/data/download.py
python src/data/preprocess.py
python src/data/augment.py                                  # ~10-20 мин (180 000 PNG)

# исходные данные
python src/models/train.py --model simple_cnn
python src/models/train.py --model resnet18
python src/models/train.py --model mobilenet_v2

# исходные + аугментированные данные
python src/models/train.py --model simple_cnn   --augmented
python src/models/train.py --model resnet18     --augmented
python src/models/train.py --model mobilenet_v2 --augmented

python src/evaluate.py    # reports/comparison.csv, comparison.md, figures/accuracy_comparison.png
```

Быстрая проверка пайплайна без полного обучения (несколько изображений, 1 эпоха):

```powershell
python src/data/download.py
python src/data/preprocess.py
python src/data/augment.py --limit 200
python src/models/train.py --model simple_cnn --limit 2000 --epochs 1 --batch_size 128
python src/evaluate.py
```

## История экспериментов

**MLflow UI** (перемещение по истории запусков, сравнение метрик, скачивание моделей):

```powershell
python -m mlflow ui --backend-store-uri ./mlruns     # http://127.0.0.1:5000
```

**DVC-эксперименты** (сравнение версий пайплайна без «грязных» коммитов):

```powershell
python -m dvc exp run -S train.lr=0.0005
python -m dvc exp show
python -m dvc exp diff
python -m dvc exp apply <exp-name>          # «откат» к конкретному эксперименту
```

## Данные в удалённом хранилище

Git хранит только `.dvc`-файлы; данные/модели — в DVC-remote:

```powershell
python -m dvc remote add -d storage <URL>    # например s3://bucket/mlops или локальная папка
git add .dvc/config
git commit -m "add dvc remote"
python -m dvc push                           # выгрузка data/ и models/ в remote
python -m dvc pull                           # восстановление на другой машине
```

## Особенности реализации

* Метки классов аугментированного датасета хранятся в папках `00..09`, поэтому
  `ImageFolder` даёт тот же порядок меток, что и FashionMNIST (проверяется в
  `train.py`, иначе обучение падает с ошибкой — иначе метрики были бы недостоверны).
* Аугментированные изображения сохраняются через `to_pil_image` (mode `L`).
  `torchvision.utils.save_image` пропускает тензор через `make_grid`, который
  разворачивает 1 канал в 3, и на диск попадал бы RGB-PNG — обучение
  одноканальных моделей падало с `expected input[...] to have 1 channels, but got 3`.
* Тест для всех экспериментов — исходный Fashion-MNIST test (10 000 изображений),
  поэтому результаты с аугментацией и без сопоставимы.
* Сплит train/val фиксирован (`data/processed/split_info.pt`, seed=42) и
  переиспользуется при обучении.
* Гиперпараметры задаются в `params.yaml`; DVC отслеживает их изменение и
  перезапускает нужные стадии.