from flask import Flask, request, render_template
import os
import json
import mysql.connector
from report_gen import generate_report

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",        # Replace with your MySQL username
    password="Adithya@2005",# Replace with your MySQL password
    database="project1"
)
cursor = db.cursor()

@app.route('/')
def index():
    return render_template('genrepo.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form inputs
        college = request.form.get('collegeName')
        event = request.form.get('eventName')
        location = request.form.get('location')
        feedback = request.form.get('feedback')

        image_files = request.files.getlist('images')
        image_paths = []
        image_filenames = []

        # Save uploaded images
        for image in image_files:
            if image.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, image.filename)
                image.save(filepath)
                image_paths.append(filepath)
                image_filenames.append(image.filename)

        # Insert into event_inputs table
        insert_input = """
            INSERT INTO event_inputs (college_name, event_name, location, feedback, images)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_input, (college, event, location, feedback, json.dumps(image_filenames)))
        db.commit()
        input_id = cursor.lastrowid  # Get the inserted input row ID

        # Generate the report
        print("\n--- Calling generate_report from app.py ---")
        report_path, report_content = generate_report(college, event, location, feedback, image_paths)

        # Debugging: Ensure report_content is not None
        print(f"Generated report content: {report_content}")
        
        # Only save the report if it was successfully generated
        if report_content:
            insert_report = "INSERT INTO event_reports (input_id, report) VALUES (%s, %s)"
            cursor.execute(insert_report, (input_id, report_content))
            db.commit()
            print(f"Report saved to event_reports with input_id {input_id}")

        if report_path:
            return f"<h3>Report generated and saved successfully!</h3><p>Saved to: {report_path}</p>"
        else:
            return "<h3>Report generation failed. Try again with better inputs.</h3>"

    except Exception as e:
        print(f"Exception in /submit route: {e}")
        return "<h3>Something went wrong. Please check your input and try again.</h3>"

if __name__ == '__main__':
    app.run(debug=True)
