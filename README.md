# Embedding-Based Face Recognition Service

A Python-based Face Recognition API built using **FastAPI** and **DeepFace**.

This service is designed to work with a **Spring Boot backend**. Python handles face detection, embedding generation, and face similarity comparison, while Spring Boot handles database operations and application/business logic.

## Architecture

```text
Frontend
   |
   v
Spring Boot Backend
   |
   |---- MySQL
   |     - Save face embeddings
   |     - Retrieve face embeddings
   |
   v
Python FastAPI Service
   |
   |---- DeepFace
   |---- FaceNet
   |
   v
Embedding Generation / Face Comparison
```

## Features

- Generate face embeddings from uploaded images
- Face recognition using embedding comparison
- Cosine similarity for face matching
- FaceNet model through DeepFace
- JPEG and PNG image support
- FastAPI-based REST endpoints
- Spring Boot integration using multipart requests
- Database-independent Python service

## Tech Stack

- Python
- FastAPI
- DeepFace
- FaceNet
- NumPy
- Uvicorn

## Project Structure

```text
FaceRecognition/
│
├── controller.py
├── service.py
├── requirements.txt
├── README.md
└── .gitignore
```

### `controller.py`

Contains the FastAPI endpoints for:

- Generating face embeddings
- Recognizing faces

### `service.py`

Contains the core face-recognition logic:

- Face embedding generation
- Cosine similarity calculation
- Embedding comparison

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI application using Uvicorn:

```bash
uvicorn controller:app --reload
```

The service will run at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Generate Face Embedding

```http
POST /generate-embedding
```

Multipart request:

```text
image: JPEG/PNG file
```

Example response:

```json
{
  "embedding": [0.123, -0.245, 0.567],
  "dimensions": 128
}
```

The generated embedding is returned to the Spring Boot application, which is responsible for storing it in the database.

---

### Recognize Face

```http
POST /recognize-face
```

Multipart request:

```text
image: JPEG/PNG file
saved_embeddings: JSON string
```

`saved_embeddings` contains face embeddings retrieved from the database by the Spring Boot application.

Example:

```json
[
  {
    "username": "teacher1",
    "embedding": [0.123, -0.245, 0.567]
  },
  {
    "username": "teacher2",
    "embedding": [0.421, -0.125, 0.331]
  }
]
```

The API generates an embedding for the uploaded image and compares it with the saved embeddings using cosine similarity.

Example response:

```json
{
  "recognized": true,
  "username": "teacher1",
  "similarity": 0.91,
  "message": "Face recognized"
}
```

If no matching face satisfies the configured threshold:

```json
{
  "recognized": false,
  "username": null,
  "similarity": 0.54,
  "message": "Unknown person"
}
```

## Face Recognition Flow

```text
Teacher Image
     |
     v
Spring Boot
     |
     v
FastAPI
     |
     v
DeepFace + FaceNet
     |
     v
Current Face Embedding
     |
     v
Cosine Similarity
     |
     +---- Saved Embeddings
     |       ^
     |       |
     |    Spring Boot
     |       |
     |      MySQL
     |
     v
Best Matching Face
     |
     v
Username + Similarity Score
```

## Responsibility Separation

### Spring Boot

Spring Boot is responsible for:

- Teacher/user management
- Saving face embeddings
- Retrieving face embeddings
- Database operations
- Attendance management
- Authentication and business logic

### Python Face Recognition Service

Python is responsible for:

- Face detection
- Face embedding generation
- Face comparison
- Cosine similarity calculation

The Python service does **not directly access the application database**.

## Current Recognition Model

The project currently uses:

```text
Model: FaceNet
Detector: OpenCV
Similarity: Cosine Similarity
```

The recognition threshold is currently configured in the Python service and should be validated/tuned using representative images from the deployment environment.

## Security

Do not commit:

- `.env` files
- Database credentials
- Virtual environments
- Teacher face images
- Generated cache files
- IDE-specific files

These should be excluded through `.gitignore`.
