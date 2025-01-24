from Cluster import Cluster  
import os

cluster_obj = Cluster(model_name="mobilenet")

image_folder = "/path/to/image/folder"

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

labels, cluster_counts = cluster_obj.create_cluster(image_paths, num_clusters=5, output_dir="clusters_output")

print("Clustering complete!")
print(f"Labels: {labels}")
print(f"Cluster Counts: {cluster_counts}")
print("Clustered images have been saved in the 'clusters_output' directory.")
