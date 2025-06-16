import os 
import sys
import pandas as pd 
import cv2

from boxmot import track



sequence_name=""
annotation_name=""
sequence_path=os.path.join("VisDrone2019-MOT-train","sequences")
annotations_path=os.path.join("VisDrone2019-MOT-train","annotations")


category_mapping={ 0: "ignored regions", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",5: "van", 6: "truck", 7  : "tricycle", 
                   8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others"}


# visualize annotations
def visualize_annotations(clip_name):
    sequence_file = os.path.join(sequence_path, clip_name  )
    annotation_file = os.path.join(annotations_path, clip_name + ".txt")

    if not os.path.exists(sequence_file):
        print(f"Sequence file {sequence_file} does not exist.")
        return

    if not os.path.exists(annotation_file):
        print(f"Annotation file {annotation_file} does not exist.")
        return

    df = pd.read_csv(annotation_file, header=None)
    df.columns = [
        "frame", "target_id", "x", "y", "w", "h",
        "score", "category", "truncation", "occlusion"
    ]
    
    df=df[df["score"]==1]
    df=df.sort_values("frame")
    for frame in sorted(df["frame"].unique()):
        print("we are in frame ",frame)
        image_path = os.path.join(sequence_file, f"{int(frame):07d}.jpg")
        if not os.path.exists(image_path):
            continue
        img = cv2.imread(image_path)
        frame_objects = df[df["frame"] == frame]   
        for _, row in frame_objects.iterrows():
            x1, y1 = int(row["x"]), int(row["y"])
            x2, y2 = x1 + int(row["w"]), y1 + int(row["h"])
            obj_id = int(row["target_id"])
            category = int(row["category"])
            label = f"ID:{obj_id} C:{category_mapping[category]}"
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow("VisDrone Annotation", img)
        key = cv2.waitKey(20)
        if key == 27:  
            break

    cv2.destroyAllWindows()




visualize_annotations("uav0000264_02760_v" )