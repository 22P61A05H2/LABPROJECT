import os
import torch
import requests
import easyocr
import warnings
from PIL import Image
from datetime import datetime
from geopy.geocoders import Nominatim
import torchvision.transforms as transforms
import torchvision.models as models
import platform
import subprocess

warnings.filterwarnings("ignore")

API_KEY = "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"  # Replace this
MODEL_NAME = "google/gemma-2-9b-it:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "EventReportGenerator"
}

reader = easyocr.Reader(['en'], gpu=False)
geolocator = Nominatim(user_agent="event_report_generator")

def process_image_text(path):
    try:
        results = reader.readtext(path)
        texts = [text for _, text, conf in results if conf > 0.6]
        return ", ".join(texts) if texts else None
    except Exception as e:
        print(f"OCR error for {path}: {e}")
        return None

def extract_image_features(image_path):
    try:
        model = models.resnet18(pretrained=True)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(image_tensor)
        return features.flatten().tolist()[:10]
    except Exception as e:
        print(f"Image feature extraction error for {image_path}: {e}")
        return []

def get_college_from_location(loc):
    try:
        if "," in loc and all(part.strip().replace('.', '', 1).isdigit() for part in loc.split(',')):
            lat, lon = map(float, loc.split(','))
            place = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        else:
            place = geolocator.geocode(loc, exactly_one=True, language='en')

        if place and 'address' in place.raw:
            addr = place.raw['address']
            return addr.get('university') or addr.get('college') or f"{addr.get('city', '')}, {addr.get('state', '')}".strip(", ")
        return loc
    except Exception as e:
        print(f"Geolocation error for '{loc}': {e}")
        return loc

def generate_prompt(data):
    event = data.get('event_name')
    college = data.get('college_name')
    location_detail = data.get('location', '')
    location_near = get_college_from_location(location_detail) if location_detail else ''
    feedback = data.get('feedback', '')
    image_texts = data.get('image_texts', [])
    image_features = data.get('image_features', [])

    if not event or not college:
        return None

    prompt = (
        f"Write a formal, well-structured institutional event report with clear and professional language in 50 words and 75% accuracy:\n\n"
        f"Event Name: {event}\n"
        f"Institution: {college}\n"
        f"Location: {location_detail} ({location_near})\n"
        f"Image Texts: {', '.join(image_texts) if image_texts else 'None'}\n"
        f"Image Features: {', '.join(map(str, image_features)) if image_features else 'None'}\n"
        f"Feedback: {feedback if feedback else 'None'}\n\n"
        f"Instructions:\n"
        f"- Use a Markdown heading '## {event} Event Report'.\n"
        f"- Describe the successful conduct of the event by the institution.\n"
        f"- Mention the event’s atmosphere, participation, and any visible themes.\n"
        f"- Include location details and relevant insights from feedback.\n"
        f"- Avoid including specific dates or club names unless provided.\n"
        f"- Ensure the report sounds like a formal institutional summary."
    )
    return prompt

def send_prompt(prompt):
    try:
        response = requests.post(API_URL, headers=HEADERS, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}]
        })
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip() if 'choices' in data else None
    except Exception as e:
        print(f"API call failed: {e}")
        return None

def check_accuracy(report, data):
    def normalize(txt): return txt.lower().strip()
    report_text = normalize(report)
    total, match = 0, 0

    for key in ['event_name', 'college_name', 'location', 'feedback']:
        val = data.get(key)
        if val:
            total += 1
            val_norm = normalize(val)
            if key == 'location':
                alt_loc = normalize(get_college_from_location(val))
                if val_norm in report_text or alt_loc in report_text:
                    match += 1
            elif key == 'feedback':
                words = val_norm.split()
                if sum(1 for w in words if w in report_text) / len(words) >= 0.4:
                    match += 1
            else:
                if val_norm in report_text:
                    match += 1
    return round(match / total, 2) if total > 0 else 1.0

def save_report(report):
    try:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {filename}")
        return os.path.abspath(filename)
    except Exception as e:
        print(f"Error saving report: {e}")
        return None

def open_report_file(path):
    if not path:
        return
    try:
        if platform.system() == 'Darwin':
            subprocess.run(['open', path])
        elif platform.system() == 'Windows':
            subprocess.run(['start', path], shell=True)
        else:
            subprocess.run(['xdg-open', path])
    except Exception as e:
        print(f"Failed to open file: {e}")

def generate_report(college_name, event_name, location, feedback, image_paths):
    try:
        image_texts = []
        image_features = []
        for path in image_paths:
            image_texts.append(process_image_text(path) or 'None')
            image_features.append(extract_image_features(path))

        data = {
            "college_name": college_name,
            "event_name": event_name,
            "location": location,
            "feedback": feedback,
            "image_texts": image_texts,
            "image_features": image_features
        }

        prompt = generate_prompt(data)
        if not prompt:
            print("Missing required input data.")
            return None

        report = send_prompt(prompt)
        if not report:
            print("Report generation failed.")
            return None

        accuracy = check_accuracy(report, data)
        print(f"Generated report with accuracy: {accuracy * 100:.1f}%")

        report_path = save_report(report)
        open_report_file(report_path)
        return report_path

    except Exception as e:
        print(f"Report generation error: {e}")
        return None
