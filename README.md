## Usando funções

### Augmentations
```
from utils import augment_dataset_with_rotations

train_dir = 'datasetv5/train'
valid_dir = 'datasetv5/valid'

print(f"Augmentation com dados de treino")
augment_dataset_with_rotations(train_dir)

print(f"\nAugmentation com dados de validação")
augment_dataset_with_rotations(valid_dir)
```

### Outros utils
#### draw_bbox
```
draw_bbox('datasetv5/train/images/1.jpg')
```

#### delete_annotations
```
delete_augmentations('datasetv5/train')
```
