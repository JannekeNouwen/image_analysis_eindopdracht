import os


def get_labels(file_path: str) -> dict[str, str]:
    """Gets the labels for every file number.

    Args:
        file_path (str): File path of the file containing the labels and the file numbers.

    Returns:
        dict[str, str]: Label for each file number.
    """
    labels = {}
    with open(file_path) as file:
        file.readline()
        for line in file:
            line = line.strip().split(",")
            if len(line) == 3:
                _, file_num, label = line
                if label:
                    labels[file_num] = label
    return labels
                        
        
def order_images(labels: dict[str, str], image_folder: str):
    """Moves files in folders for every label.

    Args:
        labels (dict[str, str]): Label for each file number.
        image_folder (str): Folder containing all image files.
    """
    for file in os.listdir(image_folder):
        if file.endswith(".jpg"):
            print(file)
            label = labels.get(str(int(file[11:16])))
            if label:
                os.rename(f"{image_folder}{file}", f"{image_folder}{label}/{file}")
                print(f"Moved {file} with label {label}")


def main():
    labels = get_labels("/home/janneke/Documents/Image analysis/eindopdracht/data/blood_cells/images/original/dataset-master/dataset-master/labels.csv")
    order_images(labels, image_folder="/home/janneke/Documents/Image analysis/eindopdracht/data/blood_cells/images/original/dataset-master/dataset-master/JPEGImages/EVAL/")
    
    
if __name__ == "__main__":
    main()