import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Set your device to use the GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the YOLO model
    model = YOLO('yolov8n.pt')  
    
    # Train the model
    model.train(data='C:/Beyblade/data.yaml', epochs=60, imgsz=640, batch=16, device=device)
