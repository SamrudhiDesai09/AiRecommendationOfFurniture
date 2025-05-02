# room_classifier.py

from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models

# Load a pre-trained MobileNet model
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()

# ImageNet index for room-like classes (simplified for demo)
ROOM_CLASSES = {
    532: 'studio_couch',
    920: 'window_shade',
    538: 'television',
    420: 'lampshade',
    511: 'refrigerator',
    759: 'wardrobe',
    746: 'table_lamp',
    829: 'sofa',
    832: 'studio_couch',
    558: 'loupe',  # might overlap with room interior
    814: 'room'    # actual room class
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

def is_room_image(image_path: str) -> bool:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = mobilenet(image_tensor)
            _, predicted = torch.max(output, 1)
            pred_idx = predicted.item()
            print(f"[DEBUG] Predicted class index: {pred_idx}")
            return pred_idx in ROOM_CLASSES
    except Exception as e:
        print(f"[ERROR] Room detection failed: {e}")
        return False
