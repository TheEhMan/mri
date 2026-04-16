# MedGemma MRI WebApp v3

This directory contains the standalone, lightweight Flask web application that serves the results of the MedGemma experimental pipeline.

It is designed to be **entirely decoupled** from the massive machine learning dependencies (PyTorch, transformers, bitsandbytes, nnUNet) required by the root pipeline. This separation allows the web app to be easily dockerized, deployed to AWS/Cloud, and run smoothly on hardware without GPUs.

## Features
- **In-Memory PNG Caching**: MRI slices are extracted from 3D NIfTI arrays dynamically on the first scroll, but cached in pure RAM as compressed PNG byte-streams subsequently. This eliminates slider lag and CPU bottlenecking over network connections.
- **Dynamic Slice Viewer**: Interactive sagittal, coronal, and axial slice rendering with built-in segmentation masking (Cost Function Masking & Tumour Projections).
- **Subtle Branding**: Clean, elegant UI with a highly-stealthy creator signature hidden in the margins.

## Requirements & Environment

If you want to run this *without* Docker, you can install the extremely lightweight web dependencies:
```bash
pip install -r webapp_v3/requirements.txt
```
*(Packages include: Flask, numpy, nibabel, pillow, and gunicorn)*

## How to Run Locally (Docker)

The easiest and most robust way to run this application is via the included Docker Compose configuration.

1. Ensure Docker Desktop is running.
2. From the *root directory* of the project (`MRI_Atlas`), run:
```bash
docker-compose up --build -d
```
3. Open a browser and navigate to **[http://localhost:5001](http://localhost:5001)**.

### Updating Data
To update the patient data inside the container, you do not need to rewrite the Flask code. Just run the data prep script from the root directory:
```bash
python webapp_v3/prepare_v3_data.py
```
After new data is copied into `webapp_v3/data` and `webapp_v3/test_output`, you must rebuild the container so the data is baked into the new image:
```bash
docker-compose up --build -d
```

## Creator 
Designed and developed by **Jamil Ur Reza**.
