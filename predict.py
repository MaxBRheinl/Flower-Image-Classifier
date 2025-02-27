import torch
from torchvision import models, transforms
from PIL import Image
import json
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
parser.add_argument('image_path', type=str, help='Path to the image')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()


# Load the checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


model = load_checkpoint(args.checkpoint)
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Image preprocessing
def process_image(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(img)


# Predict function
def predict(image_path, model, topk=5):
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze_(0).float()
    img = img.to(device)

    with torch.no_grad():
        output = model.forward(img)

    probability = torch.exp(output)
    top_probs, top_classes = probability.topk(topk, dim=1)
    top_probs = top_probs.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_classes]
    return top_probs, top_labels


# Load category names
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = None

# Make prediction
probs, classes = predict(args.image_path, model, args.top_k)

# Print results
if cat_to_name:
    labels = [cat_to_name[str(cls)] for cls in classes]
    for label, prob in zip(labels, probs):
        print(f"{label}: {prob:.3f}")
else:
    for cls, prob in zip(classes, probs):
        print(f"{cls}: {prob:.3f}")
