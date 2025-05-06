from openai import OpenAI
import re
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import requests
import json
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import speech_recognition as sr
from datetime import datetime
import easyocr  # Import the easyocr library

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
MODEL_NAME = "google/gemma-3-12b-it:free"

# Initialize easyocr reader ONCE
try:
    reader = easyocr.Reader(['en'])  # You can add other languages here, e.g., ['en', 'hi']
except Exception as e:
    print(f"Error initializing easyocr: {e}")
    reader = None

# Initialize geolocator
geolocator = Nominatim(user_agent="multimedia_report_generator")

# Load pre-trained model and labels ONCE
model = None
transform = None
imagenet_classes = []
try:
    model = models.resnet18(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        with open("imagenet_classes.txt", "r") as f:
            imagenet_classes = [s.strip() for s in f.readlines()]
    except FileNotFoundError:
        print("Warning: imagenet_classes.txt not found. Downloading...")
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            imagenet_labels_json = response.json()
            if isinstance(imagenet_labels_json, dict):
                imagenet_classes = list(imagenet_labels_json.values())
            elif isinstance(imagenet_labels_json, list):
                imagenet_classes = imagenet_labels_json
            else:
                print("Error: Unexpected format for downloaded ImageNet labels.")
                imagenet_classes = ["unknown"] * 1000
            with open("imagenet_classes.txt", "w") as f:
                for label in imagenet_classes:
                    f.write(label + "\n")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ImageNet labels: {e}")
            imagenet_classes = ["unknown"] * 1000
        except json.JSONDecodeError as e:
            print(f"Error decoding downloaded JSON: {e}")
            imagenet_classes = ["unknown"] * 1000
    except Exception as e:
        print(f"Error loading ImageNet class labels: {e}")
        imagenet_classes = ["unknown"] * 1000
except Exception as e:
    print(f"Error initializing PyTorch models or transforms: {e}")
    model = None
    transform = None
    imagenet_classes = ["Error"] * 1000

def process_image_objects(image_path):
    if model is None or transform is None or not imagenet_classes or "Error" in imagenet_classes:
        return "Error: PyTorch model or dependencies not initialized correctly."
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            output = model(batch_t)

        _, indices = torch.sort(output, descending=True)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        top5_preds = [(imagenet_classes[idx], probabilities[idx].item() * 100) for idx in indices[0][:5]]
        description = f"likely containing {', '.join([pred[0] for pred in top5_preds])}"
        return description
    except FileNotFoundError:
        return f"Error: Image file not found at '{image_path}'."
    except Exception as e:
        return f"Error processing image '{image_path}' for object detection: {e}"

def process_image_text(image_path):
    if reader is None:
        return "Error: easyocr not initialized."
    try:
        result = reader.readtext(image_path)
        if result:
            recognized_text = ", ".join([detection[1] for detection in result])
            return f"identified \"{recognized_text}\""
        else:
            return "identified no specific text"
    except FileNotFoundError:
        return f"Error: Image file not found at '{image_path}' for text recognition."
    except Exception as e:
        return f"Error processing image '{image_path}' for text recognition with easyocr: {e}"

def get_college_name_from_location(location_str):
    try:
        college = None
        if "," in location_str:
            lat_str, lon_str = location_str.split(',')
            try:
                latitude = float(lat_str.strip())
                longitude = float(lon_str.strip())
                location = geolocator.reverse((latitude, longitude), exactly_one=True, language="en")
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    college = address.get('university') or address.get('college')
            except ValueError:
                return "Error: Invalid latitude/longitude format."
        else:
            location = geolocator.geocode(location_str, exactly_one=True, language="en")
            if location and location.raw.get('address'):
                address = location.raw['address']
                college = address.get('university') or address.get('college')

        return college if college else location_str  # Return location if college not found
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        return f"Error with geocoding service: {e}"
    except Exception as e:
        return f"An unexpected error occurred during geocoding: {e}"

def generate_report_prompt(inputs):
    report_parts = []

    event_name = inputs.get('event_name', 'the event')
    college_name = inputs.get('college_name', 'the institution')
    report_parts.append(f"The {event_name} event, hosted by {college_name},")

    image_details = []
    if 'images' in inputs and inputs['images']:
        for i, image_path in enumerate(inputs['images']):
            details = []
            if inputs.get('recognize_text', False):
                text_description = process_image_text(image_path)
                details.append(f"text recognition {text_description}")
            if details:
                image_details.append(" and ".join(details))

        if image_details:
            report_parts.append(f"appears to have involved elements as indicated by {', '.join(image_details)}.")
        else:
            report_parts.append("appears to have occurred.")

    else:
        report_parts.append("appears to have occurred.")

    location_info = None
    if 'location' in inputs:
        location_str = inputs['location']
        resolved_college_name = get_college_name_from_location(location_str)
        if resolved_college_name and resolved_college_name.lower() != college_name.lower() and "[College name not found based on location]" not in resolved_college_name and "Error" not in resolved_college_name:
            location_info = f"The provided location for the event was {location_str} in {resolved_college_name}."
        else:
            location_info = f"The provided location for the event was {location_str}."
        report_parts.append(location_info)
    elif college_name != '[College Name not provided]':
        report_parts.append(f"The event was hosted at {college_name}.")

    if 'feedback' in inputs:
        report_parts.append(f"User feedback noted: \"{inputs['feedback']}\".")

    final_prompt = " ".join(report_parts)
    return final_prompt + " Please generate a detailed report summarizing these details in a natural-sounding paragraph with more than 2 or 3 paragraphs"

def generate_and_save_api_report(final_prompt):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": final_prompt}
            ]
        )

        if completion and completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            generated_report = completion.choices[0].message.content.strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_reports/api_report_{timestamp}.txt"
            try:
                with open(filename, "w") as outfile:
                    outfile.write(generated_report)
                print(f"\nAPI Generated Report saved to '{filename}'")
                return os.path.abspath(filename)  # Return the full path
            except Exception as e:
                print(f"Error saving API generated report to file: {e}")
                return None
        else:
            print("Error: Could not retrieve a valid report from the API.")
            if completion:
                print(f"Completion object: {completion}")
            else:
                print("Completion object is None.")
            return None

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

def generate_report(data):
    prompt = generate_report_prompt(data)
    print("\n--- Generated Prompt for API ---")
    print(prompt)
    return generate_and_save_api_report(prompt)

if __name__ == "__main__":
    # This block is for direct execution of report_gen.py (for testing)
    # You would typically run app.py for the web application
    inputs = {
        "event_name": "Sample Event",
        "college_name": "Sample College",
        "images": ["C:\\Users\\jyoth\\OneDrive\\Desktop\\miniproject1\\Img.jpg"],  # Replace with an actual image path for testing
        "recognize_text": True,
        "feedback": "Positive event.",
        "location": "Hyderabad"
    }
    report_path = generate_report(inputs)
    if report_path:
        print(f"Report generated and saved at: {report_path}")