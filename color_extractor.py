from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(image_path, k=3):
    image = Image.open(image_path)
    image = image.resize((150, 150))  # Resize for faster processing
    data = np.array(image)
    data = data.reshape((-1, 3))  # Flatten to RGB

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    colors = kmeans.cluster_centers_
    dominant_color = colors[0]  # First dominant color
    return tuple(map(int, dominant_color))  # Convert to RGB tuple
