import os
from tqdm import tqdm
from PIL import Image

TRAIN_PATH = "VisDrone2019-DET-train"
VAL_PATH = "VisDrone2019-DET-val"
TEST_PATH= "VisDrone2019-DET-test"
OUTPUT_PATH = "YOLO_VisDrone"

# VisDrone classes
VISDRONE_CLASSES = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others"
}

# YOLO compatible classes (we ignore class 0 and 11)
VALID_CLASSES = {k: v for k, v in VISDRONE_CLASSES.items() if k not in [0, 11]}
CLASS_ID_MAPPING = {old_id: new_id for new_id, old_id in enumerate(VALID_CLASSES)}

# For statistics
stats = {CLASS_ID_MAPPING[k]: 0 for k in VALID_CLASSES}

def convert_annotations(annotation_dir, image_dir, out_labels_dir):
    os.makedirs(out_labels_dir, exist_ok=True)

    for file in tqdm(os.listdir(annotation_dir)):
        if not file.endswith(".txt"):
            continue
        
        base_name = os.path.splitext(file)[0]
        label_output_path = os.path.join(out_labels_dir, f"{base_name}.txt")
        annotation_file = os.path.join(annotation_dir, file)

        with open(annotation_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        yolo_lines = []

        for line in lines:
            parts = line.split(',')
            if len(parts) != 8:
                print(f"Skipping {file}: Malformed line -> {line}")
                continue

            x, y, w, h = map(float, parts[:4])
            score = int(parts[4])
            category = int(parts[5])
            truncation = int(parts[6])
            occlusion = int(parts[7])

            if category not in VALID_CLASSES:
                continue

            # Convert to YOLO format
            image_path = os.path.join(image_dir, f"{base_name}.jpg")
            

            
            with Image.open(image_path) as img:
                img_w, img_h = img.size

            

            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            yolo_class_id = CLASS_ID_MAPPING[category]
            yolo_line = f"{yolo_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

            # Stats update
            stats[yolo_class_id] += 1

        if yolo_lines:
            with open(label_output_path, "w") as out_f:
                out_f.write("\n".join(yolo_lines))


# Run on train and val sets
convert_annotations(
    annotation_dir=os.path.join(TRAIN_PATH, "annotations"),
    image_dir=os.path.join(TRAIN_PATH, "images"),
    out_labels_dir=os.path.join(OUTPUT_PATH, "labels", "train")
)

convert_annotations(
    annotation_dir=os.path.join(VAL_PATH, "annotations"),
    image_dir=os.path.join(VAL_PATH, "images"),
    out_labels_dir=os.path.join(OUTPUT_PATH, "labels", "val")
)

convert_annotations(
    annotation_dir=os.path.join(TEST_PATH, "annotations"),
    image_dir=os.path.join(TEST_PATH, "images"),
    out_labels_dir=os.path.join(OUTPUT_PATH, "labels", "test")
)
# Show stats
print("\n📊 YOLO Dataset Class Distribution:")
for class_id, count in stats.items():
    class_name = list(CLASS_ID_MAPPING.keys())[list(CLASS_ID_MAPPING.values()).index(class_id)]
    print(f"Class {class_id} ({VISDRONE_CLASSES[class_name]}): {count} objects")
