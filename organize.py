import os
import shutil
import scipy.io
from sklearn.model_selection import train_test_split

data_dir = 'flowers'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    for i in range(1, 103):
        class_dir = os.path.join(split_dir, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]

image_paths = [f'jpg/image_{i:05d}.jpg' for i in range(1, len(labels) + 1)]

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
valid_paths, test_paths, valid_labels, test_labels = train_test_split(
    test_paths, test_labels, test_size=0.5, stratify=test_labels, random_state=42
)

def move_images(paths, labels, split):
    for path, label in zip(paths, labels):
        dest_dir = os.path.join(data_dir, split, str(label))
        shutil.move(path, os.path.join(dest_dir, os.path.basename(path)))

move_images(train_paths, train_labels, 'train')
move_images(valid_paths, valid_labels, 'valid')
move_images(test_paths, test_labels, 'test')

print("Dataset organized successfully.")
