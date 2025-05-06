from flask import Flask, render_template, request, send_file
from report_gen import generate_report
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def form():
    return render_template('genrepo.html') # Replace with your actual HTML file name

@app.route('/submit', methods=['POST'])
def submit():
    college = request.form.get('collegeName', '')
    event = request.form.get('eventName', '')
    location = request.form.get('location', '')
    feedback = request.form.get('feedback', '')
    images = request.files.getlist('images') # Get a list of uploaded files
    image_paths = []

    for image in images:
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                image.save(image_path)
                image_paths.append(image_path)
            except Exception as e:
                print(f"Error saving image {filename}: {e}")

    # Generate report
    print("\n--- Calling generate_report from app.py ---")
    report_path = generate_report(college=college, event=event, location=location, feedback=feedback, image_paths=image_paths)

    if report_path:
        return send_file(report_path, as_attachment=True)
    else:
        return "Error generating report.", 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)