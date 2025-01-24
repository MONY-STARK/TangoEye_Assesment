import os

def count(folder_path):

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'} 
        image_count = 0

        for files in os.listdir(folder_path):
                
                if os.path.splitext(files)[1].lower()  in image_extensions:
                        image_count+=1
        return image_count


if __name__ == "__main__":
        folder_path = r"c:\Users\Mony\Downloads\tango-cv-assessment-dataset\tango-cv-assessment-dataset"
        print(count(folder_path))   