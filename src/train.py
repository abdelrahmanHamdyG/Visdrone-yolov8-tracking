
import os
import yaml
import cv2
import torch
import random
import matplotlib.pyplot as plt 
from pathlib import PurePath
from ultralytics import YOLO
from pathlib import Path
import numpy as np

data_path = Path("path/to/your/data")  # Update with your actual data path
class_names = ["pedestrian", "people",  "bicycle","car","van","truck","tricycle", "awning-tricycle","bus","motor"]

def setup_dataset():
        """Create enhanced dataset YAML with better configuration"""
        dataset_config = {
            'path': "YOLO_VisDrone",
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names,
            
        }
        
        yaml_path = data_path / 'enhanced_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Enhanced dataset config created: {yaml_path}")
        return str(yaml_path)
    

def multi_scale_training():
        """
        Advanced multi-scale training with dynamic image sizes
        This is the KEY enhancement you suggested!
        """
        training_history = []
        dataset_yaml = setup_dataset()
        best_model_path = "C:/Users/dhamd/runs/detect/yooooolo_phase_4/weights/best.pt"

        # Multi-scale training strategy
        training_phases = [
            {
                "name": "Small Scale Foundation",
                "sizes": [384, 416], 
                "epochs": 25,
                "batch": 32,
                "lr": 0.015,
                "focus": "Learning basic features"
            },
            {
                "name": "Medium Scale Refinement", 
                "sizes": [480, 512, 544],
                "epochs": 32,
                "batch": 16, 
                "lr": 0.008,
                "focus": "Multi-scale feature learning"
            },
            {
                "name": "Large Scale Precision",
                "sizes": [608, 640, 672],
                "epochs": 25,
                "batch": 8,
                "lr": 0.004,
                "focus": "Fine-grained detection"
            },
             {
                "name": "Large Scale Precision_again",
                "sizes": [608, 640, 672],
                "epochs": 25,
                "batch": 8,
                "lr": 0.004,
                "focus": "Fine-grained detection"
            },
            {
                "name": "Large Scale Precision_again_again",
                "sizes": [608, 640, 672],
                "epochs": 20,
                "batch": 8,
                "lr": 0.004,
                "focus": "Fine-grained detection"
            }
        ]
        
        print("🎯 Starting Advanced Multi-Scale Training")
        print("=" * 60)
        
        for phase_idx, phase in enumerate(training_phases):
            print(f"\n📏 Phase {phase_idx + 1}: {phase['name']}")
            print(f"🎯 Focus: {phase['focus']}")
            print(f"📐 Image sizes: {phase['sizes']}")
            print("-" * 40)
            
            # Load model (continue from previous phase)
            if phase_idx < 4:
                continue
                model = YOLO('yolov8n.pt')
            else:
                model = YOLO(best_model_path)

            
            # Train with dynamic image sizes
            results = model.train(
                data=dataset_yaml,
                epochs=phase['epochs'],
                imgsz=phase['sizes'],  # Multiple sizes for dynamic training!
                batch=phase['batch'],
                name=f"yooooolo_phase_{phase_idx + 1}",
                device='0',
                
                # Phase-specific learning parameters
                lr0=phase['lr'],
                lrf=0.01,  # Lower final LR for better convergence
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=8.0,    
                cls=0.5,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=15.0 if phase_idx < 2 else 8.0,  # Less rotation in final phase
                translate=0.1,
                scale=0.9,
                shear=2.0,   # Add shear for more variety
                perspective=0.0001,  # Slight perspective for aerial view
                flipud=0.0,  # No vertical flip for drones
                fliplr=0.5,
                mosaic=1.0 if phase_idx < 2 else 0.5,  # Reduce mosaic in final phase
                mixup=0.15 if phase_idx < 2 else 0.05,  # Reduce mixup in final phase
                copy_paste=0.1,  # Copy-paste augmentation
                
                # Training optimization
                optimizer='AdamW',  # Often better than SGD for small objects
                close_mosaic=20,  # Close mosaic in last 10 epochs
                
                # Validation and saving
                patience=25,
                save_period=15,
                plots=True,
                val=True,
                save_json=True,
                
                # Advanced settings
                amp=True,  # Automatic Mixed Precision
                fraction=1.0,  # Use full dataset
                seed=42 + phase_idx,  # Different seed per phase
                deterministic=False,  # Allow some randomness for better generalization
            )
            
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            
            # Store phase results
            phase_metrics = {
                'phase': phase_idx + 1,
                'name': phase['name'],
                'sizes': phase['sizes'],
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0)
            }
            training_history.append(phase_metrics)
            
            print(f"✅ Phase {phase_idx + 1} Results:")
            print(f"   mAP50: {phase_metrics['mAP50']:.4f}")
            print(f"   mAP50-95: {phase_metrics['mAP50-95']:.4f}")
            print(f"   Precision: {phase_metrics['precision']:.4f}")
            print(f"   Recall: {phase_metrics['recall']:.4f}")
        
        print(f"\n🏆 Multi-scale training complete!")
        print(f"📁 Final model: {best_model_path}")
        
        return results

def test_model(model_path):
     
    dataset_yaml=setup_dataset()


    model=YOLO(model_path)

    metrics=model.val(
        data=dataset_yaml,
        split='test',
        iou=0.55,
        conf=0.25,
        save_json=True,
        save_txt=False,
        verbose=True
    )

    print("✅ Evaluation completed.")
    print("📊 mAP50: ", metrics.results_dict['metrics/mAP50(B)'])
    print("📊 mAP50-95: ", metrics.results_dict['metrics/mAP50-95(B)'])


    test_images_dir= Path("YOLO_VisDrone/images/test")
    test_labels_dir=Path("YOLO_VisDrone/labels/test")

    image_paths = list(test_images_dir.glob("*.jpg"))
    sample_images= random.sample(image_paths, 5)
     
    for image_path in sample_images:
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = model(img, conf=0.25)[0]
        pred_boxes = results.boxes

        # Draw predicted boxes
        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw ground truth boxes
        label_path = test_labels_dir / PurePath(image_path).with_suffix('.txt').name
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id, cx, cy, w, h = map(float, line.strip().split())
                    img_h, img_w = img.shape[:2]
                    cx *= img_w
                    cy *= img_h
                    w *= img_w
                    h *= img_h
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # blue for GT
                    

        # Show image with matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(f"Green: Ground Truth | Blue: predictions ")
        plt.axis('off')
        plt.show()
         

        

     



# Example usage
if __name__ == "__main__":
    # Enhanced training with all features

    results = multi_scale_training()
    # test_model("path to your model to test")