from openai import OpenAI
import os
import easyocr  # Import the easyocr library
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json  # Import the json library for debugging API response

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

def process_image_text(image_path):
    if reader is None or not image_path or not os.path.exists(image_path):
        return None
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
            try:
                latitude = float(location_str.split(',')[0].strip())
                longitude = float(location_str.split(',')[1].strip())
                location = geolocator.reverse((latitude, longitude), exactly_one=True, language="en")
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    college = address.get('university') or address.get('college')
            except ValueError:
                pass
            except GeocoderTimedOut:
                return "Error: Geocoding service timed out."
            except GeocoderServiceError as e:
                return f"Error with geocoding service: {e}"
        else:
            try:
                location = geolocator.geocode(location_str, exactly_one=True, language="en")
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    college = address.get('university') or address.get('college')
            except GeocoderTimedOut:
                return "Error: Geocoding service timed out."
            except GeocoderServiceError as e:
                return f"Error with geocoding service: {e}"
        return college if college else location_str
    except Exception as e:
        return f"An unexpected error occurred during geocoding: {e}"

def generate_report_prompt(college_name, event_name, location, feedback, image_paths):
    report_parts = []

    report_parts.append(f"The {event_name} event, hosted by {college_name},")

    image_descriptions = []
    if image_paths:
        for image_path in image_paths:
            text_description = process_image_text(image_path)
            if text_description:
                image_descriptions.append(text_description)

        if image_descriptions:
            report_parts.append(f"appears to have involved elements as {', and '.join(image_descriptions)}.")
        else:
            report_parts.append("appears to have involved visual elements.")
    else:
        report_parts.append("appears to have occurred.")

    location_info = None
    if location:
        resolved_college_name = get_college_name_from_location(location)
        if resolved_college_name and resolved_college_name.lower() != college_name.lower() and "Error" not in resolved_college_name:
            location_info = f"The provided location was {location}, possibly at or near {resolved_college_name}."
        else:
            location_info = f"The event took place at {location}."
        report_parts.append(location_info)
    elif college_name:
        report_parts.append(f"The event was hosted at {college_name}.")

    if feedback:
        report_parts.append(f"User feedback noted: \"{feedback}\".")

    final_prompt = " ".join(report_parts)
    return final_prompt + " Please generate a detailed report summarizing these details in a natural-sounding paragraph with more than 2 or 3 paragraphs."

def generate_and_save_api_report(prompt):
    print("\n--- Inside generate_and_save_api_report ---")
    print(f"Prompt being sent to API: {prompt}")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print("\n--- API Response Received ---")
        print(f"Completion object: {completion}") # Print the entire response object

        if completion and completion.choices and len(completion.choices) > 0 and completion.choices[0].message:
            generated_report = completion.choices[0].message.content.strip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("generated_reports", exist_ok=True)
            filename = f"generated_reports/api_report_{timestamp}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as outfile:
                    outfile.write(generated_report)
                print(f"\nAPI Generated Report saved to '{filename}'")
                return os.path.abspath(filename)  # Return the full path
            except Exception as e:
                print(f"Error saving API generated report to file: {e}")
                return None
        else:
            print("Error: Could not retrieve a valid report from the API.")
            if completion:
                print(f"Completion details (JSON): {json.dumps(completion.model_dump_json(), indent=2)}")
            else:
                print("Completion object is None.")
            return None

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

def generate_report(college, event, location, feedback, image_paths):
    print("\n--- Inside generate_report function ---")
    print(f"College: {college}, Event: {event}, Location: {location}, Feedback: {feedback}, Image Paths: {image_paths}")
    prompt = generate_report_prompt(college, event, location, feedback, image_paths)
    print("\n--- Generated Prompt for API ---")
    print(prompt)
    return generate_and_save_api_report(prompt)

if __name__ == "__main__":
    # This block is for direct execution of report_gen.py (for testing)
    inputs = {
        "event_name": "Test Event",
        "college_name": "Test College",
        "image_paths": ["uploads/your_image1.jpg", "uploads/your_image2.jpg"],  # List of test image paths
        "feedback": "The event was well-received.",
        "location": "Test Location"
    }
    # Create dummy uploads directory and dummy image files for testing
    os.makedirs("uploads", exist_ok=True)
    for i in range(1, 3):
        try:
            with open(f"uploads/your_image{i}.jpg", "w") as f:
                f.write("")
        except Exception as e:
            print(f"Warning: Could not create dummy test image {i}: {e}")

    report_path = generate_report(**inputs)
    if report_path:
        print(f"Report generated and saved at: {report_path}")