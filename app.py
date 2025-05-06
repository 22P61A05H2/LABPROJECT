from flask import Flask, render_template, request, send_file
from report_gen import generate_report
import os

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('genrepo.html')

@app.route('/submit', methods=['POST'])
def submit():
    college = request.form['collegeName']
    event = request.form['eventName']
    location = request.form['location']
    feedback = request.form['feedback']
    image = request.files['images']

    # Save image to disk
    image_path = os.path.join("uploads", image.filename)
    os.makedirs("uploads", exist_ok=True)
    image.save(image_path)

    # Generate report
    report_path = generate_report(college, event, location, feedback, image_path)

    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
