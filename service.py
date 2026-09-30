import os
import cv2
import numpy as np
import onnxruntime as ort



# MODEL CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobile_facenet-onnx-float",
    "mobilefacenet.onnx"
)



# CHECK MODEL FILE
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"MobileFaceNet model not found: {MODEL_PATH}"
    )



# LOAD MOBILEFACENET MODEL

# This happens ONCE when service.py is imported.
# We do NOT load the model for every request.
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# Get model input information
# ============================================================
# INSPECT ALL MODEL INPUTS AND OUTPUTS
# ============================================================

model_inputs = session.get_inputs()
model_outputs = session.get_outputs()


print("======================================")
print("MobileFaceNet loaded successfully")
print("Model:", MODEL_PATH)

print("\n===== MODEL INPUTS =====")

for inp in model_inputs:
    print(
        "Input Name:",
        inp.name,
        "| Shape:",
        inp.shape,
        "| Type:",
        inp.type
    )


print("\n===== MODEL OUTPUTS =====")

for out in model_outputs:
    print(
        "Output Name:",
        out.name,
        "| Shape:",
        out.shape,
        "| Type:",
        out.type
    )

print("======================================")


# ============================================================
# VALIDATE EXPECTED MODEL INPUT
# ============================================================

# EXPECTED_INPUT_SHAPE = [1, 3, 112, 112]

# if input_shape != EXPECTED_INPUT_SHAPE:
#     raise RuntimeError(
#         f"Unexpected model input shape. "
#         f"Expected {EXPECTED_INPUT_SHAPE}, "
#         f"but model requires {input_shape}"
#     )


# ============================================================
# LOAD OPENCV FACE DETECTOR
#
# This is our simple V1 detector.
# Later we can replace it with a better landmark-based detector.
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


if face_detector.empty():
    raise RuntimeError(
        "Failed to load OpenCV face detector."
    )


# ============================================================
# DETECT FACE
# ============================================================

def detect_face(image):
    if image is None:
        raise ValueError("Image cannot be None.")

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face detected in the image."
        )

    # --------------------------------------------------------
    # If multiple faces are detected,
    # select the largest face.
    # --------------------------------------------------------

    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Crop face
    face = image[
        y:y + h,
        x:x + w
    ]

    if face.size == 0:
        raise ValueError(
            "Detected face crop is empty."
        )

    return face


# ============================================================
# PREPROCESS FACE FOR MOBILEFACENET
#
# Model expects:
#
# [1, 3, 112, 112]
#
# N = Batch
# C = Channels
# H = Height
# W = Width
# ============================================================

def preprocess_face(face):

    if face is None or face.size == 0:
        raise ValueError(
            "Invalid face image."
        )

    # --------------------------------------------------------
    # 1. Resize
    #
    # H x W:
    # 112 x 112
    # --------------------------------------------------------

    face = cv2.resize(
        face,
        (112, 112)
    )


    # --------------------------------------------------------
    # 2. OpenCV reads:
    #
    # BGR
    #
    # MobileFaceNet uses RGB input in this pipeline.
    # --------------------------------------------------------

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # 3. Convert uint8 -> float32
    # --------------------------------------------------------

    face = face.astype(
        np.float32
    )


    # --------------------------------------------------------
    # 4. Normalize
    #
    # Original pixels:
    # 0 -> 255
    #
    # approximately converted to:
    # -1 -> +1
    #
    # NOTE:
    # Verify this against metadata.json before production.
    # --------------------------------------------------------

    face = (
        face - 127.5
    ) / 128.0


    # --------------------------------------------------------
    # Current shape:
    #
    # (112, 112, 3)
    #
    # HWC
    #
    # Convert:
    #
    # HWC -> CHW
    #
    # (3, 112, 112)
    # --------------------------------------------------------

    face = np.transpose(
        face,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Add batch dimension
    #
    # (3, 112, 112)
    #
    # becomes
    #
    # (1, 3, 112, 112)
    # --------------------------------------------------------

    face = np.expand_dims(
        face,
        axis=0
    )


    # --------------------------------------------------------
    # Ensure contiguous float32 memory.
    # Useful when passing NumPy arrays to ONNX Runtime.
    # --------------------------------------------------------

    face = np.ascontiguousarray(
        face,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if face.shape != (1, 3, 112, 112):
        raise ValueError(
            f"Invalid input tensor shape: {face.shape}"
        )

    return face


# ============================================================
# GENERATE FACE EMBEDDING
# ============================================================

# ============================================================
# GENERATE FACE EMBEDDING
# ============================================================

def generate_embedding(image_path):

    if not image_path:
        raise ValueError(
            "Image path is required."
        )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # 1. Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read image."
        )


    # --------------------------------------------------------
    # 2. Detect face
    # --------------------------------------------------------

    face = detect_face(image)


    # --------------------------------------------------------
    # 3. Preprocess face
    #
    # Output shape:
    #
    # [1, 3, 112, 112]
    # --------------------------------------------------------

    input_tensor = preprocess_face(face)


    # --------------------------------------------------------
    # 4. Run MobileFaceNet
    #
    # IMPORTANT:
    #
    # This particular ONNX model requires TWO inputs:
    #
    # img1 -> [1, 3, 112, 112]
    # img2 -> [1, 3, 112, 112]
    #
    # And returns:
    #
    # embeddings -> [2, 128]
    #
    # Because this function needs an embedding for ONE face,
    # we pass the same face tensor to both model inputs.
    # --------------------------------------------------------

    outputs = session.run(
        ["embeddings"],
        {
            "img1": input_tensor,
            "img2": input_tensor
        }
    )


    if not outputs:
        raise RuntimeError(
            "MobileFaceNet returned no output."
        )


    # --------------------------------------------------------
    # 5. Read model output
    #
    # Expected:
    #
    # embeddings.shape = (2, 128)
    #
    # embeddings[0] -> img1
    # embeddings[1] -> img2
    # --------------------------------------------------------

    embeddings = np.asarray(
        outputs[0],
        dtype=np.float32
    )


    if embeddings.shape != (2, 128):
        raise RuntimeError(
            f"Unexpected model output shape: "
            f"{embeddings.shape}. "
            f"Expected (2, 128)."
        )


    # --------------------------------------------------------
    # 6. Extract first face embedding
    #
    # Both img1 and img2 contain the same face,
    # so we only need the first result.
    # --------------------------------------------------------

    embedding = embeddings[0]


    # --------------------------------------------------------
    # 7. Validate embedding
    # --------------------------------------------------------

    if embedding.shape != (128,):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embedding.shape}"
        )


    # --------------------------------------------------------
    # 8. L2 normalization
    #
    # After normalization:
    #
    # ||embedding|| = 1
    #
    # This makes cosine similarity comparison easier.
    # --------------------------------------------------------

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(
            "Model generated a zero embedding."
        )

    embedding = embedding / norm


    # --------------------------------------------------------
    # 9. Convert NumPy array -> Python list
    #
    # Required for FastAPI JSON serialization.
    # --------------------------------------------------------

    return embedding.tolist()
# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    # Embeddings must have the same dimensions
    if a.shape != b.shape:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"{a.shape} vs {b.shape}"
        )

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    similarity = np.dot(
        a,
        b
    ) / denominator

    return float(similarity)


# ============================================================
# COMPARE CURRENT FACE AGAINST SAVED FACES
# ============================================================

def compare_embeddings(
    current_embedding,
    saved_embeddings
):

    if not saved_embeddings:
        return None, -1.0

    best_username = None
    best_score = -1.0

    for face in saved_embeddings:

        # Skip invalid records
        if "username" not in face:
            continue

        if "embedding" not in face:
            continue

        saved_embedding = face[
            "embedding"
        ]

        # Compare current face with saved face
        score = cosine_similarity(
            current_embedding,
            saved_embedding
        )

        # Keep the highest similarity
        if score > best_score:
            best_score = score
            best_username = face[
                "username"
            ]

    return (
        best_username,
        best_score
    )