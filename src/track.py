import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VisDroneTracker:
    # Your YOLOv8 class names (index matches class IDs)
    CLASS_NAMES = [
        "pedestrian", "people", "bicycle", "car", "van",
        "truck", "tricycle", "awning-tricycle", "bus", "motor"
    ]
    
    # VisDrone category mapping (0-11) - add +1 to your YOLO class IDs
    VISDRONE_CATEGORIES = {
        0: 1,  # pedestrian (0 in YOLO -> 1 in VisDrone)
        1: 2,  # people
        2: 3,  # bicycle
        3: 4,  # car
        4: 5,  # van
        5: 6,  # truck
        6: 7,  # tricycle
        7: 8,  # awning-tricycle
        8: 9,  # bus
        9: 10  # motor
    }
    
    # Categories to track (as per VisDrone challenge)
    TRACK_CATEGORIES = {1, 2, 4, 5, 6, 9}  # pedestrian, people, car, van, truck, bus

    def __init__(self, yolo_model, 
                 max_age=50, n_init=3, max_cosine_distance=0.4, nn_budget=100):
        """
        Initialize tracker with DeepSort configuration
        
        Args:
            yolo_model: Path to YOLO model weights
            max_age: Maximum frames to keep a track without detection
            n_init: Number of detections before confirming a track
            max_cosine_distance: Threshold for matching detections to tracks
            nn_budget: Maximum size for appearance descriptor gallery
        """
        self.yolo = YOLO(yolo_model)
        
        # DeepSort configuration
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget
        )
        
        # Visualization colors
        self.colors = [(np.random.randint(0, 255), 
                       np.random.randint(0, 255), 
                       np.random.randint(0, 255)) for _ in range(1000)]

    def track_video(self, video_path, output_path=None, show_live=True):
        """Track objects in a video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Process frame and get visualization
            _, visualized_frame = self._process_frame(frame, frame_idx)
            
            # Save output
            if output_path:
                out.write(visualized_frame)
            
            # Show live visualization
            if show_live:
                cv2.imshow('Tracking', visualized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def track_sequence(self, sequence_dir, output_dir=None, show_live=True):
        """Track objects in a sequence of images (like VisDrone format)"""
        frame_files = sorted([f for f in os.listdir(sequence_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx, frame_file in enumerate(frame_files, start=1):
            frame_path = os.path.join(sequence_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
                
            # Process frame and get visualization
            _, visualized_frame = self._process_frame(frame, frame_idx)
            
            # Save output
            if output_dir:
                output_path = os.path.join(output_dir, frame_file)
                cv2.imwrite(output_path, visualized_frame)
            
            # Show live visualization
            if show_live:
                cv2.imshow('Tracking', visualized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()

    def _process_frame(self, frame, frame_idx):
        """Process single frame and return (tracks, visualized_frame)"""
        # YOLO detection
        detections = self.yolo(frame, conf=0.5)[0]
        
        if detections.boxes is None or len(detections.boxes) == 0:
            return [], frame
        
        # Prepare detections for tracker
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        cls_ids = detections.boxes.cls.cpu().numpy().astype(int)
        
        dets = []
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = box
            visdrone_cls = self.VISDRONE_CATEGORIES.get(cls_id, -1)
            if visdrone_cls in self.TRACK_CATEGORIES:
                dets.append(([x1, y1, x2-x1, y2-y1], conf, visdrone_cls))
        
        # Update tracker
        tracks = self.tracker.update_tracks(dets, frame=frame)
        
        # Visualization
        vis_frame = frame.copy()
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bbox = track.to_ltrb()
            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color and draw bounding box
            color = self.colors[track_id % len(self.colors)]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return tracks, vis_frame

# Example usage
if __name__ == "__main__":
    # Initialize with your configuration
    tracker = VisDroneTracker(
        yolo_model="path/to/your/model.pt",
        max_age=25,
        n_init=3,
        max_cosine_distance=0.3,
        nn_budget=60
    )
    
    # Track a video file
    # tracker.track_video(
    #     video_path="your_video.mp4",
    #     output_path="output_video.avi",
    #     show_live=True
    # )
    
    # Or track a sequence of images
    tracker.track_sequence(
        sequence_dir="path/to/your/image_sequence",
        output_dir="output_images_folder",
        show_live=True
    )