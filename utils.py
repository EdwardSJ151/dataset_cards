import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import albumentations as A
from tqdm import tqdm

###########################################################################################
################################### Preprocessing #########################################
###########################################################################################

def read_yolo_annotations(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    # Pegar anotações dos txts
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
    return bboxes, class_labels


def write_yolo_annotations(output_label_path, bboxes, class_labels):
    # Escrever anotações dos txts
    with open(output_label_path, 'w') as f:
        for i, bbox in enumerate(bboxes):
            class_id = class_labels[i]
            x_center, y_center, width, height = bbox
            

            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            width = np.clip(width, 0.0, 1.0)
            height = np.clip(height, 0.0, 1.0)
            
            if width <= 1e-6 or height <= 1e-6:
                raise ValueError(f"Bounding box anotado incorretamente, corrija o arquivo {output_label_path}")

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def augment_dataset_with_rotations(base_dir_path, overwrite_existing=False):
    image_dir = os.path.join(base_dir_path, 'images')
    label_dir = os.path.join(base_dir_path, 'labels')
        
    # Definindo ângulos de rotação e o nome dos arquivos
    rotations = {
        "_90": 90,
        "_180": 180, 
        "_270": 270
    }
    
    # Se overwrite estiver ativado, deletar todas as imagens e labels rotacionadas
    if overwrite_existing:
        deleted_images = 0
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_part = os.path.splitext(img_file)[0]
                if any(name_part.endswith(suffix) for suffix in rotations.keys()):
                    try:
                        os.remove(os.path.join(image_dir, img_file))
                        deleted_images += 1
                    except Exception as e:
                        print(f"Erro ao deletar imagem {img_file}: {e}")
        
        deleted_labels = 0
        for label_file in os.listdir(label_dir):
            if label_file.lower().endswith('.txt'):
                name_part = os.path.splitext(label_file)[0]
                if any(name_part.endswith(suffix) for suffix in rotations.keys()):
                    try:
                        os.remove(os.path.join(label_dir, label_file))
                        deleted_labels += 1
                    except Exception as e:
                        print(f"Erro ao deletar label {label_file}: {e}")
        
        print(f"Arquivos deletados para sobrescrita:")
        print(f"  Imagens: {deleted_images}")
    
    all_images = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    original_images = []
    for img_name in all_images:
        name_part = os.path.splitext(img_name)[0]
        if not any(name_part.endswith(suffix) for suffix in rotations.keys()):
            original_images.append(img_name)
    
    if not original_images:
        print("Não há imagens para rotacionar")
        return
        
    print(f"{len(original_images)} imagens encontrados em {image_dir} para rotacionar.")
    processed_count = 0
    generated_files_count = 0
    
    # Processar as imagens
    for img_name in tqdm(original_images, desc="Processando imagens"):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, base_name + '.txt')
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_annotations(label_path)
        
        # Definindo transformação com os ângulos
        for suffix, angle in rotations.items():
            bbox_params = None
            if bboxes:
                bbox_params = A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels'],
                    min_area=1.0,
                    min_visibility=0.1
                )
            
            transform = A.Compose([
                A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT)
            ], bbox_params=bbox_params)
            
            # Aplicar transformação
            if bboxes:
                transformed = transform(
                    image=image_rgb,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                rotated_bboxes = transformed['bboxes']
                rotated_class_labels = transformed['class_labels']
            else:
                transformed = transform(image=image_rgb)
                rotated_bboxes = []
                rotated_class_labels = []
            
            rotated_image = transformed['image']
            
            # Salvar imagens e bboxes
            new_img_name = f"{base_name}{suffix}.jpg"
            new_img_path = os.path.join(image_dir, new_img_name)
            cv2.imwrite(new_img_path, cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR))
            generated_files_count += 1
            
            if rotated_bboxes:
                new_label_path = os.path.join(label_dir, f"{base_name}{suffix}.txt")
                write_yolo_annotations(new_label_path, rotated_bboxes, rotated_class_labels)
                generated_files_count += 1
        
        processed_count += 1

    print(f"{processed_count} imagens processadas.")
    print(f"{generated_files_count} novos arquivos foram gerados (images and labels).")


###########################################################################################
################################## Other Utils ############################################
###########################################################################################

def draw_bbox(image_path, img_size=None, show_bbox_image=False):
    if not os.path.exists(image_path):
        print(f"Imagem não encontrada: {image_path}")
        return
    
    label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    
    parts = image_path.split(os.sep)
    try:
        split_idx = max([i for i, part in enumerate(parts) if part in ['train', 'valid']])
        dataset_root = os.sep.join(parts[:split_idx])
        yaml_path = os.path.join(dataset_root, 'data.yaml')
    except:
        yaml_path = None
    
    class_names = {}
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
            names_section = False
            for line in lines:
                line = line.strip()
                if line == 'names:':
                    names_section = True
                    continue
                if names_section and line and not line.startswith('#'):
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            try:
                                idx = int(parts[0].strip())
                                name = parts[1].strip()
                                class_names[idx] = name
                            except ValueError:
                                continue
    
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_height, original_width, _ = original_image.shape
    
    if img_size:
        try:
            target_width, target_height = map(int, img_size.split('x'))
            image = cv2.resize(original_image, (target_width, target_height))
            width, height = target_width, target_height
            width_scale = target_width / original_width
            height_scale = target_height / original_height
        except (ValueError, AttributeError):
            print(f"Formato de img_size inválido. Use 'LARGURAxALTURA', por exemplo: '416x416'")
            image = original_image
            height, width = original_height, original_width
            width_scale = height_scale = 1.0
    else:
        image = original_image
        height, width = original_height, original_width
        width_scale = height_scale = 1.0
    
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = line.split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        box_width = float(values[3])
                        box_height = float(values[4])
                        
                        x_min = int((x_center - box_width/2) * width)
                        y_min = int((y_center - box_height/2) * height)
                        box_width_px = int(box_width * width)
                        box_height_px = int(box_height * height)
                        
                        bboxes.append({
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, f"Class {class_id}"),
                            'x_min': x_min,
                            'y_min': y_min,
                            'width': box_width_px,
                            'height': box_height_px
                        })
    else:
        print(f"Arquivo de labels não encontrado: {label_path}")
    
    if show_bbox_image and bboxes:
        fig = plt.figure(figsize=(15, 10))
        
        main_ax = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
        main_ax.imshow(image)
        
        n_bboxes = len(bboxes)
        cols = min(4, n_bboxes)
        rows = (n_bboxes + cols - 1) // cols
        
        gs = fig.add_gridspec(3, 4)
        crop_axes = []
        for i in range(min(n_bboxes, 4)):
            crop_axes.append(fig.add_subplot(gs[2, i]))
    else:
        plt.figure(figsize=(12, 10))
        plt.imshow(image)
        main_ax = plt.gca()
    
    np.random.seed(42)
    colors = np.random.rand(100, 3)
    
    for i, bbox in enumerate(bboxes):
        color = colors[bbox['class_id'] % len(colors)]
        
        rect = Rectangle(
            (bbox['x_min'], bbox['y_min']),
            bbox['width'], bbox['height'],
            linewidth=2, 
            edgecolor=color,
            facecolor='none'
        )
        main_ax.add_patch(rect)
        
        size_text = f"{bbox['width']}x{bbox['height']}"
        main_ax.text(
            bbox['x_min'], 
            bbox['y_min'] - 5, 
            f"{bbox['class_name']} ({size_text})",
            color='white', 
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, pad=2)
        )
        
        if show_bbox_image and i < 4:
            x_min, y_min = max(0, bbox['x_min']), max(0, bbox['y_min'])
            x_max = min(width, x_min + bbox['width'])
            y_max = min(height, y_min + bbox['height'])
            
            crop = image[y_min:y_max, x_min:x_max]
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            
            crop_axes[i].imshow(crop)
            crop_axes[i].set_title(f"{bbox['class_name']} ({crop_width}x{crop_height})")
            crop_axes[i].axis('off')
    
    main_ax.set_title(f"Image: {os.path.basename(image_path)}" + 
                      (f" (Resized to {width}x{height})" if img_size else ""))
    main_ax.axis('off')
    plt.tight_layout()
    plt.show()

def delete_augmentations(base_dir_path):

    image_dir = os.path.join(base_dir_path, 'images')
    label_dir = os.path.join(base_dir_path, 'labels')
    
    suffixes = ["_90", "_180", "_270"]
    
    deleted_images = 0
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_part = os.path.splitext(img_file)[0]
            if any(name_part.endswith(suffix) for suffix in suffixes):
                try:
                    os.remove(os.path.join(image_dir, img_file))
                    deleted_images += 1
                except Exception as e:
                    print(f"Erro ao deletar imagem {img_file}: {e}")
    
    deleted_labels = 0
    for label_file in os.listdir(label_dir):
        if label_file.lower().endswith('.txt'):
            name_part = os.path.splitext(label_file)[0]
            if any(name_part.endswith(suffix) for suffix in suffixes):
                try:
                    os.remove(os.path.join(label_dir, label_file))
                    deleted_labels += 1
                except Exception as e:
                    print(f"Erro ao deletar label {label_file}: {e}")
    
    print(f"Arquivos deletados em {base_dir_path}")