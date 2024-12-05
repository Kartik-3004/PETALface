import os
import csv

data_root = '/mnt/store/knaraya4/data/ijbs_aligned_180'
output_csv = '/mnt/store/knaraya4/data/IJBS/image_paths_180.csv'

image_paths = []

for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, file))

with open(output_csv, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['index', 'path']) 
    for index, path in enumerate(image_paths):
        csv_writer.writerow([index, path])

print(f'All image paths have been written to {output_csv}')