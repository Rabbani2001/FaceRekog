from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import tempfile
import json
import os

from service import generate_embedding, compare_embeddings


app = FastAPI(title="Face Recognition API")

@app.get("/greet")
async def greet():
    print("Hello from server1")
    return {"message": "Hello from server1"}
# ============================================================
# GENERATE EMBEDDING
# Java sends image
# Python returns embedding
# Java saves embedding in database
# ============================================================

@app.post("/generate-embedding")
async def create_embedding(
    image: UploadFile = File(...)
):

    temp_path = None

    try:

        # Validate image type
        if image.content_type not in (
            "image/jpeg",
            "image/png"
        ):
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are allowed."
            )

        # Read image
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Image is empty."
            )

        # Get extension
        suffix = os.path.splitext(image.filename or "")[1]

        if suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png"
        ):
            suffix = ".jpg"

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # Generate embedding
        embedding = generate_embedding(temp_path)

        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# RECOGNIZE FACE
#
# Java sends:
# 1. image
# 2. saved_embeddings
#
# Python:
# 1. generates embedding from image
# 2. compares against saved embeddings
# 3. returns best matching username
# ============================================================

@app.post("/recognize-face")
async def recognize_face(
    image: UploadFile = File(...),
    saved_embeddings: str = Form(...)
):

    temp_path = None

    try:

        # Validate image
        if image.content_type not in (
            "image/jpeg",
            "image/png"
        ):
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images are allowed."
            )

        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Image is empty."
            )

        # Convert JSON string from Java -> Python list
        try:

            saved_faces = json.loads(
                saved_embeddings
            )

        except json.JSONDecodeError:

            raise HTTPException(
                status_code=400,
                detail="Invalid saved__embeddings JSON."
            )


        if not saved_faces:

            raise HTTPException(
                status_code=404,
                detail="No saved face embeddings found."
            )


        # Prepare image
        suffix = os.path.splitext(
            image.filename or ""
        )[1]

        if suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png"
        ):
            suffix = ".jpg"


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # Generate embedding for current image
        current_embedding = generate_embedding(
            temp_path
        )

      
        # Compare with embeddings received from Java
        username, score = compare_embeddings(current_embedding,saved_faces)


        if username is None:
            raise HTTPException(
                status_code=404,
                detail="No embeddings available for comparison.")

        # Recognition threshold
        THRESHOLD = 0.60
        if score < THRESHOLD:

            return {
                "recognized": False,
                "username": None,
                "similarity": round(
                    float(score),
                    4
                ),
                "message": "Unknown person"
            }

        return {
            "recognized": True,
            "username": username,
            "similarity": round(
                float(score),
                4
            ),
            "message": "Face recognized"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Face recognition failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)