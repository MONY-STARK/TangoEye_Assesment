from Retrieval import ImageRetrieval
import os

retrieval = ImageRetrieval(model_name="mobilenet")

image_folder = "/path/to/image/folder"
query_image_path = "/path/to/query/image.jpg" 

image_paths = [
    os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))
]

retrieval.extract_embeddings(image_paths)

query_embedding = retrieval.extract_query_embedding(query_image_path)

similar_images = retrieval.retrieve_similar_images(query_embedding, top_n=5)

print(f"Top 5 similar images for the query image {query_image_path}:")
for idx, (image_path, similarity) in enumerate(similar_images):
    print(f"{idx + 1}. Image: {image_path}, Similarity: {similarity:.4f}")

retrieval.display_images(query_image_path, similar_images)
