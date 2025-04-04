import json
from transformers import AutoProcessor, AutoModel
import requests
from pathlib import Path
from notebook_utils import collect_telemetry
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from megadetector.utils import url_utils
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector
import cv2
import os
import plotly.express as px


all_labels = ['badger', 'bat', 'bird', 'bobcat', 'car', 'cat', 'cow', 'coyote', 'deer', 'dog', 'empty', 'fox', 'insect', 'lizard', 'mountain_lion', 'opossum', 'pig', 'rabbit', 'raccoon', 'rodent', 'skunk', 'squirrel']
print(all_labels)
number_of_labels = len(all_labels)


# Load the SigLIP model and processor
if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)


collect_telemetry("siglip-zero-shot-image-classification.ipynb")


model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Load the MegaDetector model
detector_model = run_detector.load_detector('MDV5B')


def predict_image_label(image, model, processor, all_labels):
    if image is None:
        return "empty"
    
    text_descriptions = [f"This is a camera trap image of a {label}" for label in all_labels]
    inputs = processor(text=text_descriptions, images=[image], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        model.config.torchscript = False
        results = model(**inputs)
    logits_per_image = results["logits_per_image"]
    probs = logits_per_image.softmax(dim=1).detach().numpy()
    predicted_label_index = np.argmax(probs[0])
    predicted_label = all_labels[predicted_label_index]
    return predicted_label, max(probs[0])


def crop_image_with_megadetector(image, detector_model):
    """
    Function to crop the image using MegaDetector and return the first cropped image or None if no detection.
    params:
      image: input image
      detector_model: loaded MegaDetector model
    returns:
      cropped_image: the first cropped image or None if no detection
    """
    result = detector_model.generate_detections_one_image(image)
    detections_above_threshold = [d for d in result['detections'] if d['conf'] > 0.2]
    
    if detections_above_threshold:
        cropped_images = vis_utils.crop_image(detections_above_threshold, image, confidence_threshold=0.2, expansion=10)
        return cropped_images[0] if cropped_images else None
    else:
        return None
    
def process_video(video, detector_model, model, processor, all_labels):
    """
    Function to process a video, detect animals in each frame using MegaDetector, and classify the detected animals.
    params:
      video: video file object
      detector_model: loaded MegaDetector model
      model: loaded image classification model
      processor: processor for the image classification model
      all_labels: list of all possible labels
    returns:
      detected_labels: list of detected labels for each frame
    """
    detected_labels = []

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Total number of frames in the video: {total_frames}, FPS: {fps}")

    i = 0

    while video.isOpened():
        i += 1

        # Process only 1/10 of the frames
        ret, frame = video.read()
        if not ret:
            break
        if i % 10 != 0:
            continue
        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Crop the image using MegaDetector
        cropped_image = crop_image_with_megadetector(image, detector_model)

        if cropped_image:
            # Predict the label
            predicted_label, prob = predict_image_label(cropped_image, model, processor, all_labels)
            if not any(label[0] == predicted_label for label in detected_labels):
                detected_labels.append((predicted_label, prob))
            else:
                index = next((index for index, label in enumerate(detected_labels) if label[0] == predicted_label), None)
                detected_labels[index] = (predicted_label, max(prob, detected_labels[index][1]))

    # Release the video capture object
    video.release()
    return detected_labels

def plot_detected_labels(detected_labels):
  """
  Function to plot the detected labels and their probabilities using Plotly Express.
  params:
    detected_labels: list of tuples containing detected labels and their probabilities
  returns:
    None
  """
  labels, probs = zip(*detected_labels)
  probs = np.array(probs)
  
  top_labels = np.argsort(-probs)[: min(5, probs.shape[0])]
  top_probs = probs[top_labels]
  top_labels = [labels[index] for index in top_labels]

  if len(top_labels) > 4:
      top_labels = top_labels[:4]
      top_probs = top_probs[:4]
  
  fig = px.bar(
      x=top_probs,
      y=top_labels,
      orientation='h',
      labels={'x': 'Probability', 'y': 'Labels'},
      title='Detected Labels and Probabilities'
  )
  
  fig.update_layout(yaxis={'categoryorder': 'total ascending'})

  return fig
