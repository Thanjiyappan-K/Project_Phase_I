# 📝 Automated Detection of Copied Handwritten Submissions-TEAM 164 🚀

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/Hugging%20Face-TrOCR-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)

## 📌 Overview

Academic integrity is crucial, but traditional plagiarism checkers (like Turnitin) cannot read **handwritten** assignments. This project bridges that gap.

This is an **End-to-End Automated Pipeline** that ingests scanned PDF submissions, preprocesses them using Computer Vision, extracts text using the state-of-the-art **TrOCR (Transformer-based OCR)** model, and algorithmically compares every student's work against every other student to detect verbatim copying.

## 🚀 Key Features

* **📄 Batch Processing:** Automatically converts and processes folders full of PDF submissions.
* **🖼️ Advanced Preprocessing:** Uses OpenCV for Denoising and Adaptive Thresholding to clean noisy scans.
* **🤖 AI-Powered OCR:** Utilizes `microsoft/trocr-large-handwritten` for superior accuracy on diverse handwriting styles.
* **🔍 Pairwise Comparison:** Compares every document against every other document ($N^2$ complexity) to ensure no copy goes unnoticed.
* **📊 Visual Reports:** Generates a similarity matrix heatmap and a detailed text report flagging suspicious pairs (Threshold > 75%).

---

## 🛠️ System Architecture

The pipeline follows a linear flow:

1.  **Input:** Batch of Student PDFs.
2.  **Conversion:** PDF $\rightarrow$ High-Res Images.
3.  **Preprocessing:** Grayscale $\rightarrow$ Bilateral Filter $\rightarrow$ Adaptive Thresholding.
4.  **Inference:** TrOCR Encoder-Decoder Model extracts text.
5.  **Normalization:** Lowercase conversion and whitespace removal.
6.  **Analysis:** Sequence Matching (Gestalt Pattern Matching).
7.  **Output:** Plagiarism Report & Heatmap.

---

## ⚙️ Installation

### Prerequisites
* Python 3.8+
* **Poppler** (Required for `pdf2image`):
    * *Windows:* Download binary and add to PATH.
    * *Linux:* `sudo apt-get install poppler-utils`
    * *Mac:* `brew install poppler`

### Install Dependencies
Run the following command to install the required Python libraries:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python
pip install pdf2image
pip install scikit-learn
pip install seaborn
pip install matplotlib
pip install pillow
```

## 💻 UsageClone the Repository:
Bash git clone
```
https://github.com/Prakashmathi15/AUTOMATED-DETECTION-OF-COPIED-HANDWRITTEN-SUBMISSIONS-USING-AI-BASED-OCR-AND-SIMILARITY-ANALYSIS.git
cd handwritten-plagiarism-detector
```
## Prepare Input:     
Place all student PDF files in the student_submissions/ directory.
## Run the Pipeline:  
Bashpython main.py   
## Sample code: 
```python
# ✅ Step 1: Install required library
!pip install PyPDF2 tqdm


# ✅ Step 2: Import libraries
import os
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm
from google.colab import files


# ✅ Step 3: Upload your PDFs
print("📂 Upload your PDF files (you can select multiple files at once)")
uploaded = files.upload()


# ✅ Step 4: Create output writer
merged_writer = PdfWriter()


# ✅ Step 5: Process each uploaded PDF
for filename in tqdm(uploaded.keys(), desc="Processing PDFs"):
    reader = PdfReader(filename)
    writer = PdfWriter()


    # Extract first 4 pages (or less if fewer pages)
    for i in range(min(4, len(reader.pages))):
        merged_writer.add_page(reader.pages[i])


# ✅ Step 6: Write the combined result
output_filename = "merged_first_4_pages.pdf"
with open(output_filename, "wb") as output_file:
    merged_writer.write(output_file)


print(f"\n✅ Merging complete! File saved as: {output_filename}")


# ✅ Step 7: Download the merged PDF
files.download(output_filename)

```

## 1. INSTALL INFERENCE DEPENDENCIES (Simplified)
```python
!pip install -q "transformers[torch]" "pillow" "opencv-python-headless" "numpy"

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import warnings
import os # Import os to check if file exists


warnings.filterwarnings("ignore", message="The channel dimension is ambiguous.*")


```
##  2. LOAD FINE-TUNED MODEL & PROCESSOR  

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROCESSOR_ID = "microsoft/trocr-base-handwritten"
MODEL_ID = "Thanjiyappankanniyappan/trocr-iam-finetuned"


print(f"Loading Processor from: {PROCESSOR_ID}")
print(f"Loading Model from: {MODEL_ID} (using {DEVICE})")


try:
    processor = TrOCRProcessor.from_pretrained(PROCESSOR_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.config.max_length = 256
    print("Model and Processor loaded successfully.")


except Exception as e:
    print(f"Error loading model: {e}")
    raise

```
## 3. HELPER FUNCTION: DESKEW IMAGE      

```python
def deskew_image(image: np.ndarray) -> np.ndarray:
    """Deskews an image by finding the minimum bounding rectangle of the text."""
    # 1. Convert to grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 2. Find all non-zero points (text pixels)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image # No text found


    # 3. Get the minimum area bounding box
    rect = cv2.minAreaRect(coords)
    angle = rect[2]


    # Angle correction logic
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle


    # Don't rotate if the angle is negligible
    if abs(angle) < 0.1:
        return image


    # 4. Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)


    # Use BORDER_REPLICATE to fill in empty space, avoiding black bars
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


    return deskewed

```
## 4. GEOMETRIC LINE SEGMENTATION (STRIP EXTRACTION)      
```python
def segment_lines_from_page(page_image_np: np.ndarray) -> list[np.ndarray]:
    """
    It uses Connected Component Analysis to find each line.
    """
    # 1. Deskew the image first
    deskewed_image = deskew_image(page_image_np)


    # 2. Convert to grayscale and binarize
    gray = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 3. Dilate HORIZONTALLY to connect words into solid lines
    kernel = np.ones((3, 100), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)


    # 4. Find Contours (the outlines of the blobs)
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 5. Get the bounding box for each contour and sort them
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)


        # Filter out noise
        if w > 50 and h > 10: # Only keep if width > 50 and height > 10
            bounding_boxes.append((y, x, w, h))


    # Sort the boxes by their Y-coordinate (top-to-bottom)
    bounding_boxes.sort(key=lambda box: box[0])


    # 6. Create the actual image crops (strips)
    line_images = []
    for (y, x, w, h) in bounding_boxes:
        padding_y = int(h * 0.15) # 15% vertical padding
        y_top = max(0, y - padding_y)
        y_bot = min(deskewed_image.shape[0], y + h + padding_y)


        # Crop from the *deskewed original* BGR image
        line_crop = deskewed_image[y_top:y_bot, :]
        line_images.append(line_crop)


    return line_images

def recognize_line(line_crop_np: np.ndarray) -> str:
    """
    Takes a single line crop (numpy array) and runs it through
    the loaded TrOCR model, returning the recognized text.
    """


    # 1. Convert from OpenCV/numpy format (BGR) to PIL Image (RGB)
    try:
        rgb_crop = cv2.cvtColor(line_crop_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
    except Exception as e:
        return "[Error: Bad Crop]"


    # 2. Process image with TrOCR processor
    pixel_values = processor(
        images=pil_image,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)


    # 3. Generate text tokens
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=model.config.max_length)


    # 4. Decode tokens into a string
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

IMAGE_PATH = "/content/lines/line_15.png"


print(f"\nProcessing Page from '{IMAGE_PATH}'...")


# 1. Load the page from the JPEG file
try:
    # Check if file exists
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Please make sure 'test.jpeg' is uploaded.")


    # Read the image directly with OpenCV
    page_image_np = cv2.imread(IMAGE_PATH)


    if page_image_np is None:
        raise IOError(f"Could not read image from {IMAGE_PATH}. File may be corrupt.")


    print("JPEG page loaded successfully.")


except Exception as e:
    print(f"Error loading JPEG: {e}")
    raise


# 2. Segment the page into lines
print("Segmenting page into lines...")
line_crops = segment_lines_from_page(page_image_np)
print(f"Found {len(line_crops)} potential lines of text.")


# 3. Run OCR on each line
print("\n--- ✍️ EXTRACTED TEXT ---")
full_page_text = []
for i, crop in enumerate(line_crops):
    text = recognize_line(crop)
    print(f"{i+1:02d}: {text}")
    full_page_text.append(text)


print("\n--- 📄 Reconstructed Page Text ---")
print("\n".join(full_page_text))
print("--- End of Inference ---")
```
## View Results:
Check the console for the text report. 
Open similarity_matrix.png to see the heatmap.
Check logs/plagiarism_report.txt for the detailed breakdown.
## 📸 Sample Outputs1. 
## Preprocessing Pipeline

   ![4](https://github.com/user-attachments/assets/0993d143-fb60-47c7-b8b8-499cf96cddbb)     
    
   ![3](https://github.com/user-attachments/assets/e8a75459-2c29-48c6-8ea7-c68caad0cc88)

   ![1](https://github.com/user-attachments/assets/daf9296f-6a29-4696-a015-8c93ec352969)


## 🧠 Algorithms Used  
ComponentAlgorithm/ModelPurposeOCRTrOCR (Vision Transformer)End-to-end text recognition from images.          
PreprocessingAdaptive Gaussian ThresholdingHandling uneven lighting and shadows in scans.                
ComparisonSequenceMatcherFinding longest contiguous matching subsequences.              
## 🔮 Future Scope

Semantic Analysis: Integrating BERT/RoBERTa to detect paraphrased content (copied ideas, not just words).   

Web Interface: Developing a React/Flask web app for easy drag-and-drop usage.       

Internet Search: Extending the check to compare against online sources (Wikipedia, Chegg) via APIs.
## 👥 Contributors
  1.) PRAKASH M - CSE(CS)              
  2.) THANJIYAPPAN K - AI&ML          
  3.) SANJAY K - CSE             
## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
