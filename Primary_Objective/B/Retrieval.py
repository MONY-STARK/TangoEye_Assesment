import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from A.Cluster import Cluster  # Import your Cluster class

class ImageRetrieval:
    def __init__(self, model_name="mobilenet"):
        self.cluster_obj = Cluster(model_name=model_name)
        self.all_embeddings = []
        self.all_image_paths = []

    def extract_embeddings(self, image_paths):
        """
        Extract embeddings for all images in the given paths.
        """
        for image_path in tqdm(image_paths, desc="Extracting Embeddings", unit="image"):
            try:
                embedding = self.cluster_obj.__extract_embedding(image_path)
                self.all_embeddings.append(embedding)
                self.all_image_paths.append(image_path)
            except Exception as e:
                print(f"Error extracting embedding for {image_path}: {e}")

        self.all_embeddings = np.array(self.all_embeddings)
        print(f"Extracted embeddings for {len(self.all_embeddings)} images.")

    def extract_query_embedding(self, query_image_path):
        """
        Extract the embedding for a query image.
        """
        try:
            return self.cluster_obj.__extract_embedding(query_image_path)
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            return None

    def retrieve_similar_images(self, query_embedding, top_n=5):
        """
        Retrieve the top N most similar images to the query image.
        """
        if query_embedding is None or len(self.all_embeddings) == 0:
            print("Embeddings or query embedding not available.")
            return []

        similarities = np.dot(self.all_embeddings, query_embedding) / (
            np.linalg.norm(self.all_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[::-1][:top_n]
        similar_images = [(self.all_image_paths[i], similarities[i]) for i in top_indices]
        return similar_images

    @staticmethod
    def display_images(query_image_path, similar_images):
        """
        Display the query image and the top similar images.
        """
        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(similar_images) + 1, 1)
        plt.imshow(query_image)
        plt.axis("off")
        plt.title("Query Image")

        for idx, (image_path, similarity) in enumerate(similar_images):
            similar_image = cv2.imread(image_path)
            similar_image = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(similar_images) + 1, idx + 2)
            plt.imshow(similar_image)
            plt.axis("off")
            plt.title(f"Sim: {similarity:.4f}")

        plt.tight_layout()
        plt.show()

   