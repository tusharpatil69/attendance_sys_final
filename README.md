# Face Recognition Attendance System

This project is a **Face Recognition Attendance System** that uses **Flask** for the backend and integrates facial recognition using **MTCNN** for face detection and **FaceNet** for face embeddings. It allows you to register users, recognize faces in real-time, and log attendance in an **Excel file**.

## Features
1. User Registration:
   - Capture images for a new user via webcam.
   - Generate and store facial embeddings for the user.
   - Save user details in a SQLite database.

2. Real-Time Face Recognition:
   - Detect and recognize faces in real-time via webcam.
   - Log attendance in an Excel file (`Excel1.xlsx`).

3. Data Persistence:
   - Store user information in a SQLite database.
   - Store facial embeddings in an NPZ file for fast loading.
   - Log attendance in an Excel file for easy review.

4. Flask Web Interface:
   - Simple UI to start detection and add users.

---

## Prerequisites

### Install Dependencies
- Python 3.7+ is required.
- Install the required libraries using the following command:
  ```bash
  pip install -r requirements.txt
  ```
  Ensure the `requirements.txt` includes:
  ```plaintext
  Flask
  opencv-python
  numpy
  torch
  facenet-pytorch
  openpyxl
  sqlite3 (built-in with Python)
  ```

### Hardware
- A webcam is required for face capture and real-time detection.


## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/tusharpatil69/attendance_sys_final
cd attendance_sys_final
```

### 2. Initialize Database and Files
The following files will be created or initialized automatically when the project starts:
- `attendance_system.db`: SQLite database for user details.
- `embeddings.npz`: Stores facial embeddings and corresponding user names.
- `Excel1.xlsx`: Excel file to log attendance.

### 3. Run the Flask App
Start the Flask server with:
```bash
python app.py
```

The application will be available at `http://0.0.0.0:5000`.

---

## Usage

### 1. Home Page
Visit the home page at `http://localhost:5000`. 

### 2. Add a New User
1. Send a `POST` request to `/add_user` with the user's name:
   - Using an API testing tool (e.g., Postman) or HTML form.
   - Example form input:
     ```json
     {
       "user_name": "Tushar"
     }
     ```
2. The system will capture 5 images via webcam, generate embeddings, and store them in the database.

### 3. Start Face Recognition
1. Send a `POST` request to `/start_detection`.
2. The system will open the webcam, detect and recognize faces in real-time.
3. Attendance will be logged in `Excel1.xlsx`.

---

## File Structure

```plaintext
.
├── app.py                # Main application code
├── embeddings.npz        # Facial embeddings (generated automatically)
├── attendance_system.db  # SQLite database (generated automatically)
├── Excel1.xlsx           # Attendance log (generated automatically)
├── templates/
│   └── index.html        # HTML template for the home page
├── images/               # Folder to store user images
└── requirements.txt      # List of required dependencies
```

---

## How It Works

1. **Adding Users**:
   - The webcam captures multiple images of the user.
   - MTCNN detects the face, and FaceNet generates embeddings.
   - The embeddings are saved in `embeddings.npz`, and user data is stored in `attendance_system.db`.

2. **Recognizing Faces**:
   - The system compares webcam face embeddings to stored embeddings.
   - If a match is found (distance < threshold), the user's attendance is logged.

3. **Attendance Logging**:
   - The attendance is logged in `Excel1.xlsx` with the following columns:
     - User Name
     - Date
     - Time



## Notes

- To avoid duplicate entries, the system checks if a user’s attendance is already logged for the current date.
- Facial recognition threshold can be adjusted by modifying the distance comparison in the `perform_face_recognition` function (default: `0.75`).

---

## Future Enhancements
- Add a web-based front-end for managing users and viewing attendance records.
- Integrate with cloud services for real-time data syncing.
- Add support for multiple cameras or video streams.

---

## Troubleshooting

### 1. Common Errors
- **Camera not accessible**:
  Ensure your webcam is connected and not being used by another application.
- **MTCNN detection issues**:
  Make sure `facenet-pytorch` is correctly installed.

### 2. Debugging
Run the app in debug mode for detailed error logs:
```bash
python app.py --debug
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [FaceNet](https://github.com/timesler/facenet-pytorch) for the pre-trained model.
- OpenCV for image processing.
- Flask for building the backend.
