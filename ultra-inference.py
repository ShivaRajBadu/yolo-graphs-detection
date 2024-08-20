import json
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


url = "https://api.ultralytics.com/v1/predict/vpDf4sEEU0Fg3AiAbQVC"
headers = {"x-api-key": ""}
data = {"imgsz": 640, "conf": 0.25, "iou": 0.45}

image_path = "/home/badu/workspace/learn/scripts/data-clean/test6.webp"  # Update this with your image path

# Open the image file
with open(image_path, "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"file": f})

# Check for successful response
response.raise_for_status()

# Parse the response
results = response.json()
print(json.dumps(results, indent=2))

# Load the image
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# Define colors for different labels (you can customize these)
colors = {
    'bar_chart': 'orange',

    'graph': 'yellow',
  
    'pie_chart': 'red',
 
 
}

font_size=30
font = ImageFont.load_default()     

# Extract bounding box information
for result in results['images'][0]['results']:
    box = result['box']
    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    class_name = result['name']
    confidence = result['confidence']
    
    # Choose color based on class name
    color = colors.get(class_name, "red")  # Default to red if class not found
    
    # Draw the bounding box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Add label and confidence with larger text
    label = f"{class_name} ({confidence:.2f})"
    draw.text((x1, y1 - font_size), label, font=font, fill=color)

# Show the image with the bounding box and labels
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()

