# from PIL import Image
# import torchvision.transforms as transforms
# import torch
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# import urllib.request

# # Load model with weights
# weights = MobileNet_V2_Weights.DEFAULT
# mobilenet = mobilenet_v2(weights=weights)
# mobilenet.eval()

# # Download ImageNet class labels
# LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# with urllib.request.urlopen(LABELS_URL) as f:
#     imagenet_classes = [line.decode('utf-8').strip() for line in f.readlines()]

# # Keywords to detect room-like scenes
# ROOM_KEYWORDS = [
#     'room', 'interior', 'bedroom', 'living room', 'dining room',
#     'hall', 'studio', 'indoor', 'lounge', 'apartment', 'office',
#     'ceiling', 'floor', 'wall', 'furniture', 'workspace', 'cubicle',
#     'desk', 'chair', 'carpet', 'curtain', 'window', 'light', 'lamp'
# ]

# # Transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# def is_room_image(image_path: str, confidence_threshold=0.02, topk=8) -> bool:
#     """
#     Classify image as room if any top-k predictions match room-related keywords.
#     """
#     try:
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = transform(image).unsqueeze(0)

#         with torch.no_grad():
#             outputs = mobilenet(image_tensor)
#             probs = torch.nn.functional.softmax(outputs[0], dim=0)

#         top_probs, top_indices = torch.topk(probs, topk)

#         print("\n[DEBUG] Predictions:")
#         for i in range(topk):
#             class_name = imagenet_classes[top_indices[i]]
#             prob = top_probs[i].item()
#             print(f" - {class_name}: {prob:.4f}")
#             if prob >= confidence_threshold:
#                 for keyword in ROOM_KEYWORDS:
#                     if keyword in class_name.lower():
#                         return True

#         return False

#     except Exception as e:
#         print(f"[ERROR] Room detection failed: {e}")
#         return False


from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import urllib.request

# Load model with weights
weights = MobileNet_V2_Weights.DEFAULT
mobilenet = mobilenet_v2(weights=weights)
mobilenet.eval()

# Download ImageNet class labels
# LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# with urllib.request.urlopen(LABELS_URL) as f:
#     imagenet_classes = [line.decode('utf-8').strip() for line in f.readlines()]

with open("assets/imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Expanded keywords to detect room-like scenes or common furniture items
ROOM_KEYWORDS = [
    'room', 'interior', 'bedroom', 'living room', 'dining room',
    'hall', 'studio', 'indoor', 'apartment', 'office', 'workspace',
    'ceiling', 'floor', 'wall', 'furniture', 'window', 'curtain',
    'desk', 'chair', 'sofa', 'couch', 'lamp', 'light', 'shelf',
    'bookshelf', 'wardrobe', 'cabinet', 'mirror', 'table', 'rug', 'carpet', 'tv', 'television'
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def is_room_image(image_path: str, confidence_threshold=0.01, topk=10) -> bool:
    """
    Classify image as room if any top-k predictions match room-related keywords.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = mobilenet(image_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top_probs, top_indices = torch.topk(probs, topk)

        print("\n[DEBUG] Room classification predictions:")
        for i in range(topk):
            class_index = top_indices[i].item()
            class_name = imagenet_classes[class_index]
            prob = top_probs[i].item()
            print(f" - {class_name} ({prob:.4f})")

            # Looser match for more object names typical in rooms
            if prob >= confidence_threshold:
                for keyword in ROOM_KEYWORDS:
                    if keyword in class_name.lower():
                        print(f"Matched keyword: '{keyword}' in class '{class_name}'")
                        return True

        return False

    except Exception as e:
        print(f"[ERROR] Room detection failed: {e}")
        return False
