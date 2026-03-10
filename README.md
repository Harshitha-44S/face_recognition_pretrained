# Face Recognition System 👥

A web-based face recognition application built with Streamlit that uses pre-trained face recognition models to identify and classify faces in images.

## Features

- Real-time face detection and recognition
- Support for webcam input
- Image upload and processing
- Face comparison and matching with trained model
- Confidence scoring for face matches
- Automatic tracking of known faces and unknown persons
- Interactive web dashboard

## Project Structure

```
face_recognition_pretrained/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Project dependencies
├── face_recognizer_20260307_070311.pkl # Trained face recognition model
├── README.md                           # This file
├── static/
│   └── uploads/                        # Uploaded images directory
├── templates/
│   ├── index.html                      # Main page template
│   └── dashboard.html                  # Dashboard template
└── venv_py39/                          # Python virtual environment
```

## Requirements

- Python 3.9 or higher
- pip (Python package manager)

## Installation

### 1. Clone or Download the Project
```bash
cd face_recognition_pretrained
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Using Python 3.9
python -m venv venv_py39
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv_py39\Scripts\activate
```

**macOS/Linux:**
```bash
source venv_py39/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following Python packages:

- **streamlit** - Web application framework
- **numpy** - Numerical computing
- **opencv-python** - Computer vision library
- **pillow** - Image processing
- **pandas** - Data manipulation
- **face-recognition** - Face detection and recognition (pre-trained models)

See `requirements.txt` for specific versions.

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### How to Use

1. **Upload an Image**: Use the file uploader to select an image containing faces
2. **Webcam Input**: Optionally use your webcam for real-time face recognition
3. **View Results**: The application will:
   - Detect all faces in the image
   - Compare them against known faces in the model
   - Display classification (Team Member, Other Person, or Unknown)
   - Show confidence scores for each match

## Model Details

- **Model File**: `face_recognizer_20260307_070311.pkl`
- **Format**: Pickled Python object
- **Contains**: Pre-trained face encodings and associated metadata
- **Face Matching**: Uses Euclidean distance for face comparison
- **Confidence Threshold**: Adjustable tolerance parameter (default: 0.5)

## How It Works

1. **Face Detection**: Uses the face_recognition library to detect faces in images
2. **Face Encoding**: Converts detected faces into 128-dimensional encodings
3. **Face Comparison**: Compares input face encodings with pre-trained encodings in the model
4. **Classification**: Determines if the face matches a known person or is unknown
5. **Confidence Scoring**: Calculates confidence based on Euclidean distance

## Configuration

### Adjusting Tolerance
To change face matching sensitivity, modify the `tolerance` parameter in the code:
```python
recognizer.recognize(image_path, tolerance=0.5)  # Lower = stricter matching
```

## API Response Codes

- `TEAM_MEMBER_[name]` - Face identified as a team member
- `OTHER_PERSON_[name]` - Face identified as a known non-team member
- `UNKNOWN_PERSON` - Face doesn't match any known faces
- `NO_FACE` - No face detected in the image
- `NO_KNOWN_FACES` - Model has no trained faces loaded
- `ERROR` - Error occurred during processing

## Troubleshooting

### Model Not Loading
- Ensure `face_recognizer_20260307_070311.pkl` is in the project root
- Check file permissions
- Verify pickle file integrity

### No Faces Detected
- Ensure image quality is good
- Face must be clearly visible
- Try different images or camera positions

### Poor Recognition Accuracy
- Increase the tolerance value for more lenient matching
- Add more training samples to the model
- Ensure lighting conditions are adequate

## Future Enhancements

- [ ] Real-time video stream processing
- [ ] Adding new faces to the model
- [ ] Database integration for face records
- [ ] Performance metrics and logging
- [ ] Multi-face batch processing
- [ ] Export results to CSV/PDF

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions, please check:
1. Model file path and permissions
2. Dependencies installation
3. Python version compatibility (3.9+)
4. Image format and quality

---

**Last Updated**: March 2026  
**Python Version**: 3.9+  
**Status**: Active Development
