import os
import pandas as pd

def load_data(csv_path, directory_path):
    df = pd.read_csv(csv_path)
    image_paths = []
    labels = []
    for _, row in df.iterrows():
        folder_path = os.path.join(directory_path, row['Directory'])
        label = [row['Labels_1'], row['Labels_2']]
        for img_filename in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, img_filename))
            labels.append(label)
    return image_paths, labels

def count_images(csv_path, directory_path):
    df = pd.read_csv(csv_path)
    total_images = 0
    for directory in df['Directory'].unique():
        folder_path = os.path.join(directory_path, directory)
        total_images += len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    return total_images
