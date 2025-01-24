import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from models import Person_Re_ID_Model


class Cluster:
    def __init__(self, model_name="mobilenet"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Person_Re_ID_Model().choose_model(model_name=model_name).to(self.device)
        self.model.eval()

    def __preprocess_image(self, image_path):
        if not os.path.isfile(image_path):
            raise ValueError(f"Provided path {image_path} is not a valid file.")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded. Please check the file path.")

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor()
        ])
        tensor = preprocess(img).unsqueeze(0).to(self.device)
        return tensor

    def __extract_embedding(self, image_path):
        input_tensor = self.__preprocess_image(image_path)
        with torch.no_grad():
            embedding = self.model(input_tensor)
        return embedding.cpu().numpy().flatten()

    def create_cluster(self, image_paths, num_clusters=10, output_dir=None):
        if not isinstance(image_paths, list):
            raise ValueError("image_paths must be a list of file paths.")

        embeddings = []
        valid_paths = []

        for path in image_paths:
            try:
                embedding = self.__extract_embedding(path)
                embeddings.append(embedding)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if len(embeddings) == 0:
            raise ValueError("No valid embeddings were extracted.")

        embeddings = np.array(embeddings)
        embeddings_normalized = normalize(embeddings, axis=1)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings_normalized)

        unique_labels = set(labels)
        print(f"Clusters formed: {len(unique_labels)}")
        cluster_counts = {label: (labels == label).sum() for label in unique_labels}
        print(f"Cluster counts: {cluster_counts}")

        if output_dir:
            self.__save_clustered_images(labels, valid_paths, output_dir)

        return labels, cluster_counts

    def __save_clustered_images(self, labels, image_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for label in set(labels):
            cluster_dir = os.path.join(output_dir, f"cluster_{label}")
            os.makedirs(cluster_dir, exist_ok=True)

            for idx, image_path in enumerate(image_paths):
                if labels[idx] == label:
                    filename = os.path.basename(image_path)
                    destination = os.path.join(cluster_dir, filename)
                    cv2.imwrite(destination, cv2.imread(image_path))
        print(f"Clustered images saved to {output_dir}")


    