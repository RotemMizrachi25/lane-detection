# Lane and Vehicle Detection System

## About
This project is developed as part of the Computer Vision course. It implements a Python-based system for real-time lane and vehicle detection, adaptable to various scenarios including daytime, nighttime, and roads with crosswalks. The system features lane change detection and vehicle proximity alerts, enhancing safety across different driving conditions.

## Features
**Daytime Lane Detection:** Utilizes Canny edge detection to identify lane markings. The system applies a polygonal mask to isolate the region of interest and uses thresholding techniques for clear day conditions.\
**Nighttime Lane Detection:** Enhances lane visibility under low light conditions using adaptive thresholding techniques. This feature adjusts the detection parameters to accommodate reduced visibility and headlight glare.\
**Crosswalk Detection:** Employs specific region masking and dynamic threshold adjustments to detect crosswalks, enhancing pedestrian safety in urban environments.\
**Vehicle Proximity Detection:** Implements object detection in designated areas of the video frame to identify vehicles that are too close for safety. This feature uses contour detection to highlight vehicles based on size and location criteria.\
**Lane Change Detection:** Analyzes the continuity and orientation of detected lane lines to alert drivers about potential lane changes. This function warns drivers if lane changes are detected based on lane boundary deviations.\

## Project Structure
**main.py:** The core script that integrates all detection functionalities, including lane and vehicle detection for various conditions. Users can select different detection modes directly through command-line arguments.

## Prerequisites
Ensure you have the following installed on your system:\

Python 3.8 or later\
OpenCV 4.5 or later\
NumPy

## Getting Started
Clone the Repository:
```bash
git clone https://github.com/your-username/your-repository.git
#Install Required Dependencies
pip install opencv-python numpy
#Run the Detection System
python main.py
