import os
import shutil
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import numpy as np


def augment_character_images_pil():

    source_dir = 'TangutRecognitionDataset'  
    target_dir = 'TangutRecognitionDataset_augmented'  


    for phase in ['train', 'val']:
        phase_source_dir = os.path.join(source_dir, phase)
        phase_target_dir = os.path.join(target_dir, phase)

        for structure in ['S', 'V', 'H', 'E']:
            structure_source_dir = os.path.join(phase_source_dir, structure)

            if not os.path.exists(structure_source_dir):
                continue


            for char_id in os.listdir(structure_source_dir):
                char_source_dir = os.path.join(structure_source_dir, char_id)
                char_target_dir = os.path.join(phase_target_dir, structure, char_id)

                if not os.path.isdir(char_source_dir):
                    continue

                os.makedirs(char_target_dir, exist_ok=True)

                original_images = [f for f in os.listdir(char_source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

                if not original_images:
                    continue

                for img_name in original_images:
                    source_path = os.path.join(char_source_dir, img_name)
                    target_path = os.path.join(char_target_dir, img_name)

                    img = Image.open(source_path)
                    img.save(target_path)

                if phase == 'train':
                    augmentation_factor = 5  

                    for img_name in original_images:
                        source_path = os.path.join(char_source_dir, img_name)

                        original_img = Image.open(source_path).convert('RGB')

                        for i in range(augmentation_factor):
                            try:
                                augmented_img = apply_random_augmentation(original_img)

                                aug_img_name = f"aug_{i}_{img_name}"
                                aug_target_path = os.path.join(char_target_dir, aug_img_name)

                                augmented_img.save(aug_target_path)

                            except Exception as e:
                                print(f"    增强失败 {img_name}_{i}: {e}")
                                continue

    print_stats(target_dir)


def apply_random_augmentation(image):
    img = image.copy()

    augmentations = random.sample([
        'rotate', 'scale', 'brightness', 'contrast',
        'sharpness', 'blur', 'elastic'
    ], random.randint(1, 3))

    for aug_type in augmentations:
        if aug_type == 'rotate' and random.random() < 0.6:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

        elif aug_type == 'scale' and random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            if scale != 1.0:
                img = img.resize((100, 100), Image.Resampling.LANCZOS)

        elif aug_type == 'brightness' and random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        elif aug_type == 'contrast' and random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        elif aug_type == 'sharpness' and random.random() < 0.4:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.5))

        elif aug_type == 'blur' and random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        elif aug_type == 'elastic' and random.random() < 0.2:
            img = apply_elastic_transform(img)

    return img


def apply_elastic_transform(image, alpha=20, sigma=5):
    image_array = np.array(image)
    shape = image_array.shape

    dx = np.random.uniform(-1, 1, shape[:2]) * alpha
    dy = np.random.uniform(-1, 1, shape[:2]) * alpha

    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

    from scipy.ndimage import map_coordinates
    transformed = np.zeros_like(image_array)
    for channel in range(shape[2]):
        transformed[:, :, channel] = map_coordinates(image_array[:, :, channel],
                                                     [indices_y, indices_x],
                                                     order=1, mode='reflect')

    return Image.fromarray(transformed.astype(np.uint8))


def print_stats(dataset_dir):

    for phase in ['train', 'val']:
        print(f"\n{phase}集:")
        phase_dir = os.path.join(dataset_dir, phase)
        for structure in ['S', 'V', 'H', 'E']:
            structure_dir = os.path.join(phase_dir, structure)
            if os.path.exists(structure_dir):
                char_folders = [d for d in os.listdir(structure_dir) if os.path.isdir(os.path.join(structure_dir, d))]
                total_samples = 0
                for char_folder in char_folders:
                    char_path = os.path.join(structure_dir, char_folder)
                    samples = len([f for f in os.listdir(char_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    total_samples += samples

                print(f"  {structure}: {len(char_folders)}个字符, {total_samples}个样本")


if __name__ == "__main__":
    augment_character_images_pil()