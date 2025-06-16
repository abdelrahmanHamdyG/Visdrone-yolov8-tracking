import cv2
import os
import numpy as np
import motmetrics as mm
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VisDroneEvaluator:
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
    
    # Categories to evaluate (as per VisDrone challenge)
    EVAL_CATEGORIES = {1, 2, 4, 5, 6, 9}  # pedestrian, people, car, van, truck, bus

    def __init__(self, yolo_model, dataset_root,
                 max_age=50, n_init=3, max_cosine_distance=0.4, nn_budget=100,
                 track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6):
        """
        Initialize evaluator with extended DeepSort configuration
        
        Args:
            yolo_model: Path to YOLO model weights
            dataset_root: Root directory of VisDrone dataset
            max_age: Maximum frames to keep a track without detection
            n_init: Number of detections before confirming a track
            max_cosine_distance: Threshold for matching detections to tracks
            nn_budget: Maximum size for appearance descriptor gallery
            track_high_thresh: Detection confidence threshold
            track_low_thresh: Additional threshold for tentative tracks
            new_track_thresh: New track confidence threshold
        """
        self.yolo = YOLO(yolo_model)
        self.dataset_root = dataset_root
        
        # Extended DeepSort configuration
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            # track_high_thresh=track_high_thresh,
            # track_low_thresh=track_low_thresh,
            # new_track_thresh=new_track_thresh
        )
        
        # Setup paths
        self.test_seq_dir = os.path.join(dataset_root, 'sequences')
        self.test_anno_dir = os.path.join(dataset_root, 'annotations')
        self.output_dir = os.path.join('results', 'deepsort')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Visualization colors
        self.colors = [(np.random.randint(0, 255), 
                       np.random.randint(0, 255), 
                       np.random.randint(0, 255)) for _ in range(1000)]

    def evaluate_test_set(self, show_live=False, save_results=True):
        """Evaluate on entire VisDrone test set with visualization options"""
        seq_names = sorted([d for d in os.listdir(self.test_seq_dir) 
                          if os.path.isdir(os.path.join(self.test_seq_dir, d))])
        
        all_metrics = []
        
        for seq_name in seq_names:
            print(f"\nProcessing sequence: {seq_name}")
            seq_metrics = self.process_sequence(seq_name, show_live, save_results)
            all_metrics.append(seq_metrics)
        
        # Compute overall metrics
        if all_metrics:
            final_metrics = pd.concat(all_metrics).mean()
            self._print_final_metrics(final_metrics)
            return final_metrics
        return None

    def process_sequence(self, seq_name, show_live=False, save_results=True):
        """Process and evaluate a single sequence"""
        seq_path = os.path.join(self.test_seq_dir, seq_name)
        anno_path = os.path.join(self.test_anno_dir, f"{seq_name}.txt")
        output_path = os.path.join(self.output_dir, f"{seq_name}.txt")
        
        # Initialize metrics
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Load ground truth
        gt_data = self._load_visdrone_file(anno_path)
        
        # Process frames
        frame_files = sorted([f for f in os.listdir(seq_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for frame_idx, frame_file in enumerate(frame_files, start=1):
            frame_path = os.path.join(seq_path, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
                
            # Process frame and get tracks
            tracks, visualized_frame = self._process_frame(frame, frame_idx)
            
            # Update MOT metrics
            self._update_mot_metrics(acc, gt_data, tracks, frame_idx)
            
            # Save results
            if save_results:
                self._save_tracks(tracks, frame_idx, output_path)
            
            # Show live visualization
            if show_live:
                cv2.imshow('Tracking', visualized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if show_live:
            cv2.destroyAllWindows()
        
        # Compute metrics for this sequence
        return self._compute_metrics(acc)

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
            if visdrone_cls in self.EVAL_CATEGORIES:
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

    def _update_mot_metrics(self, acc, gt_data, tracks, frame_idx):
        """Update MOT metrics accumulator"""
        gt_frame = gt_data[gt_data['frame_id'] == frame_idx]
        gt_frame = gt_frame[gt_frame['object_category'].isin(self.EVAL_CATEGORIES)]
        gt_frame = gt_frame[gt_frame['score'] == 1]
        
        gt_boxes = gt_frame[['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height']].values
        gt_ids = gt_frame['target_id'].values
        
        if len(gt_boxes) > 0:
            gt_boxes[:, 2:] += gt_boxes[:, :2]
        
        track_boxes = []
        track_ids = []
        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_ltrb()
                track_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                track_ids.append(track.track_id)
        
        if len(gt_boxes) > 0 and len(track_boxes) > 0:
            dist_matrix = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)
        else:
            dist_matrix = np.empty((len(gt_boxes), len(track_boxes)))
        
        acc.update(gt_ids, track_ids, dist_matrix)

    def _save_tracks(self, tracks, frame_idx, output_path):
        """Save tracks in VisDrone format"""
        with open(output_path, 'a') as f:
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                bbox = track.to_ltrb()
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                
                line = f"{frame_idx},{track.track_id},{x1},{y1},{w},{h},1,{track.det_class},-1,-1\n"
                f.write(line)

    def _compute_metrics(self, acc):
        """Compute MOT metrics from accumulator"""
        mh = mm.metrics.create()
        return mh.compute(acc, metrics=['mota', 'motp', 'precision', 'recall', 'idf1'], name='overall')

    def _load_visdrone_file(self, file_path):
        """Load VisDrone format file"""
        columns = [
            'frame_id', 'target_id', 'bbox_left', 'bbox_top', 
            'bbox_width', 'bbox_height', 'score', 'object_category', 
            'truncation', 'occlusion'
        ]
        try:
            return pd.read_csv(file_path, header=None, names=columns)
        except:
            return pd.DataFrame()

    def _print_final_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\nFinal Evaluation Metrics:")
        print(f"MOTA: {metrics['mota']*100:.2f}%")
        print(f"MOTP: {metrics['motp']*100:.2f}%") 
        print(f"IDF1: {metrics['idf1']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")

# Example usage
if __name__ == "__main__":
    # Initialize with your configuration
    evaluator = VisDroneEvaluator(
        yolo_model="path/to/your/model",
        dataset_root="VisDrone2019-MOT-test-dev",
        max_age=25,
        n_init=3,
        max_cosine_distance=0.3,
        nn_budget=60,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6
    )
    
    # Evaluate with live visualization
    evaluator.evaluate_test_set(show_live=False, save_results=True)