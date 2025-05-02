
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from color_extractor import extract_dominant_color

# Load dataset
df = pd.read_csv('data/furnitures.csv')

# Preprocess color vectors (RGB)
df['color_vector'] = df[['color_r', 'color_g', 'color_b']].apply(
    lambda row: np.array([row['color_r'], row['color_g'], row['color_b']]),
    axis=1
)

def recommend_furniture(uploaded_image_path: str, category: str, top_n: int = 4) -> list:
    try:
        # Step 1: Extract dominant color
        dominant_color = extract_dominant_color(uploaded_image_path)
        query_vector = np.array(dominant_color).reshape(1, -1)

        # Step 2: Filter dataset by category (case insensitive)
        category_df = df[df['name'].str.lower().str.contains(category.lower())].copy()

        print(f"[DEBUG] Category selected: {category}, Matching items: {len(category_df)}")

        if category_df.empty:
            print("[INFO] No furniture found for this category.")
            return []

        # Step 3: Compute similarity
        category_vectors = np.stack(category_df['color_vector'].to_numpy())
        similarity_scores = cosine_similarity(query_vector, category_vectors)[0]

        # Step 4: Sort and select top results
        top_indices = similarity_scores.argsort()[::-1][:min(top_n, len(category_df))]
        recommended_images = category_df.iloc[top_indices]['image_url'].tolist()

        print(f"[INFO] Recommended {len(recommended_images)} items.")
        return recommended_images

    except Exception as e:
        print(f"[ERROR] Recommendation failed: {e}")
        return []
