import os
folder_path = 'database'
images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
images.sort(reverse=True)  # Sort images by latest first
img = "result/" + images[0]
print(img)
    