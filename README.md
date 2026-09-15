# 🧠 Brain Tumor AI — Full-Stack MRI Classification & Explainable AI

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-232F3E?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2/)

A production-ready, full-stack AI platform for multi-class brain tumor classification from MRI scans. Features a custom **TensorFlow/Keras CNN**, **LIME Explainable AI (XAI)** interpretability layer, **FastAPI REST backend**, interactive **Nginx-served frontend**, persistent **SQLAlchemy authentication**, and an **AWS EC2 / Docker** deployment pipeline.

> ⚠️ **Medical Disclaimer:** Intended strictly for research and portfolio demonstration. Not a certified clinical diagnostic tool.

---

## 💡 Quick Overview

| Domain | Key Highlights |
| :--- | :--- |
| **Machine Learning** | Custom CNN, Data Augmentation, Multi-class Softmax classification (Glioma, Meningioma, Pituitary, No Tumor) |
| **Explainable AI (XAI)** | LIME (Local Interpretable Model-agnostic Explanations) for region-of-interest heatmaps |
| **Backend & Database** | FastAPI REST API, Automatic OpenAPI/Swagger documentation, SQLAlchemy ORM with persistent SQLite |
| **Frontend UI** | Responsive Vanilla JS Dashboard, real-time image preview, probability visualizer, auth system |
| **DevOps & Cloud** | Docker multi-container setup (Nginx + FastAPI), Docker Hub publishing, AWS EC2 cloud infrastructure |

---

## 🏗️ System Architecture

```text
               ┌─────────────────────────────────────────┐
               │           Client Browser                │
               │   HTML5 / CSS3 / Vanilla JavaScript    │
               └────────────────────┬────────────────────┘
                                    │
                             HTTP   │ Port 8080
                                    ▼
               ┌─────────────────────────────────────────┐
               │         Frontend Container              │
               │            Nginx Alpine                 │
               └────────────────────┬────────────────────┘
                                    │
                         REST API   │ Port 8000
                                    ▼
               ┌─────────────────────────────────────────┐
               │          Backend Container              │
               │               FastAPI                   │
               │  ┌──────────────────┬────────────────┐  │
               │  │  TensorFlow CNN  │  LIME Engine   │  │
               │  └──────────────────┴────────────────┘  │
               │                     │                   │
               │         SQLAlchemy + SQLite             │
               └─────────────────────┬───────────────────┘
                                     │
                                     ▼
                             AWS EC2 Instance
---

## 🔄 Application Workflow

The application follows an end-to-end workflow from user authentication to MRI classification and explainable prediction.

```text
┌─────────────────────┐
│      User           │
│   Opens Web App     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Register / Login    │
│   Authentication    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Upload MRI Image  │
│  Preview MRI Scan   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Frontend JavaScript│
│   Sends API Request │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    FastAPI Backend  │
│      /predict       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Image Preprocessing │
│ Resize + Normalize  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TensorFlow / Keras  │
│     CNN Model       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│     Classification Result   │
│                             │
│  Glioma                    │
│  Meningioma                │
│  No Tumor                  │
│  Pituitary                 │
└──────────┬──────────────────┘
           │
           ├──────────────────────┐
           │                      │
           ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│ Class Probabilities │  │    LIME Analysis    │
│    Visualization    │  │ Explain Prediction  │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           └────────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │   Results Dashboard │
              │                     │
              │ • Prediction        │
              │ • Probabilities     │
              │ • LIME Explanation  │
              └─────────────────────┘

---
## 🧠 Machine Learning Pipeline

The application uses a TensorFlow/Keras CNN to classify brain MRI scans into four categories: **Glioma, Meningioma, Pituitary, and No Tumor**.

```text
MRI Image → Resize (224×224) → Normalize (0–1)
          → Data Augmentation → CNN
          → Softmax Probabilities → Predicted Class


## 🔍 Explainable AI — LIME

To improve model interpretability, the application integrates **LIME (Local Interpretable Model-Agnostic Explanations)**.

After generating a prediction, the system can analyze the MRI image and identify regions that contributed most to the model's decision.

```text
MRI Image
    │
    ▼
CNN Prediction
    │
    ▼
LIME Analysis
    │
    ▼
Important Image Regions
    │
    ▼
Visual Explanation


## ⚡ FastAPI Backend

The application uses **FastAPI** to provide a lightweight and scalable REST API for authentication, MRI prediction, and explainable AI.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API health/status check |
| `POST` | `/register` | Register a new user |
| `POST` | `/login` | Authenticate an existing user |
| `POST` | `/predict` | Classify a brain MRI image |
| `POST` | `/explain` | Generate a LIME explanation |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Backend Responsibilities

- User registration and authentication
- MRI image upload handling
- Brain tumor classification using the trained CNN
- Prediction probability generation
- LIME-based visual explanations
- SQLite database integration using SQLAlchemy
- REST API communication with the frontend

FastAPI's built-in **Swagger UI** makes it easy to test and explore the available endpoints during development and deployment.



## 🔐 Database & Authentication

The application includes a simple user authentication system backed by **SQLite** and **SQLAlchemy**.

### Authentication Flow

```text
User Registration
       │
       ▼
   FastAPI API
       │
       ▼
 SQLite Database
       │
       ▼
 User Account Created


### For User Login

 User Login
    │
    ▼
FastAPI Authentication
    │
    ▼
Validate User Credentials
    │
    ▼
Access Dashboard
                               
## 🎨 Frontend

The frontend provides the user-facing interface for authentication, MRI image upload, prediction results, and explainable AI visualization.

### Features

- User registration and login
- Brain MRI image upload
- Real-time communication with the FastAPI backend
- Predicted tumor class and probability display
- LIME explanation visualization
- Dashboard-based user experience
- Responsive interface using HTML and CSS

### Technology

- **HTML5**
- **CSS3**
- **JavaScript**
- **Nginx** for serving the frontend in Docker

The frontend dynamically determines the backend API host from the current browser hostname, allowing the same frontend build to work across local Docker and AWS EC2 deployments.



## 🐳 Docker Architecture

The application is containerized using **Docker**, with separate containers for the backend API and frontend web server.

### Container Architecture

```text
                    ┌──────────────────────────┐
                    │        User / Browser     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Frontend Container     │
                    │   Nginx + HTML/CSS/JS    │
                    │        Port: 8080        │
                    └────────────┬─────────────┘
                                 │
                                 │ HTTP API Requests
                                 ▼
                    ┌──────────────────────────┐
                    │   Backend Container      │
                    │   FastAPI + TensorFlow   │
                    │        Port: 8000        │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    ▼                          ▼
          ┌─────────────────┐       ┌──────────────────┐
          │  SQLite Database │       │  CNN Model       │
          │ brain_tumor.db   │       │ best_model.keras │
          └─────────────────┘       └──────────────────┘


## 📦 Docker Hub

The containerized application is published to **Docker Hub** so the deployment environment can pull pre-built images without rebuilding the application on the server.

### Image Workflow

```text
Source Code
     │
     ▼
Build Docker Images
     │
     ▼
Push Images to Docker Hub
     │
     ▼
AWS EC2
     │
     ▼
Pull Images
     │
     ▼
Run Containers



## ☁️ AWS EC2 Deployment

The application is deployed on **Amazon EC2** using Docker containers.

### Deployment Environment

- **Cloud Provider:** AWS
- **Service:** Amazon EC2
- **Operating System:** Ubuntu Server
- **Instance Type:** `t3.small`
- **Architecture:** x86_64
- **Storage:** 20 GiB gp3
- **Region:** Asia Pacific (Mumbai)

### Deployment Architecture

```text
                    AWS EC2 Instance
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
     ┌─────────────────┐   ┌─────────────────┐
     │ Backend Docker  │   │ Frontend Docker │
     │    Container    │   │    Container    │
     │    Port 8000    │   │    Port 8080    │
     └────────┬────────┘   └─────────────────┘
              │
       ┌──────┴───────┐
       ▼              ▼
  SQLite Database   CNN Model



## 📈 Project Evolution

The project evolved from a standalone CNN-based classification model into a complete full-stack AI application.

### Evolution

```text
CNN Model Development
        │
        ▼
Brain MRI Classification
        │
        ▼
Model Evaluation
        │
        ▼
FastAPI Inference Backend
        │
        ▼
LIME Explainability
        │
        ▼
User Authentication + Database
        │
        ▼
Web-Based Frontend
        │
        ▼
Docker Containerization
        │
        ▼
Docker Hub Image Distribution
        │
        ▼
AWS EC2 Deployment



## 🛠️ Engineering Highlights

This project demonstrates practical software engineering and deployment skills alongside machine learning.

### Key Engineering Practices

- Designed a **RESTful backend** using FastAPI
- Integrated a trained **TensorFlow/Keras CNN** into an inference API
- Added **LIME-based explainability** for model predictions
- Implemented **SQLite + SQLAlchemy** for persistent application data
- Built a separate **frontend and backend architecture**
- Containerized services using **Docker**
- Used **Docker Compose** to orchestrate multiple containers
- Published Docker images to **Docker Hub**
- Deployed the application on **AWS EC2**
- Configured AWS **Security Groups** for application access
- Persisted the SQLite database outside the backend container
- Used `.gitignore` and `.dockerignore` to prevent unnecessary files and sensitive/local artifacts from being included in source control or Docker builds
- Maintained the original machine learning work while extending it into a deployable full-stack application



## 🎯 Project Purpose

This project was developed to explore how a machine learning model can be transformed into a complete, deployable AI application.

Rather than stopping at model training and evaluation, the project combines:

- 🧠 **Machine Learning** — CNN-based MRI classification
- 🔍 **Explainable AI** — LIME-based visual explanations
- ⚡ **Backend Development** — FastAPI REST API
- 🎨 **Frontend Development** — HTML, CSS, and JavaScript
- 🔐 **Authentication** — User registration and login
- 🗄️ **Database** — SQLite with SQLAlchemy
- 🐳 **Containerization** — Docker and Docker Compose
- 📦 **Image Distribution** — Docker Hub
- ☁️ **Cloud Deployment** — AWS EC2

The result is an end-to-end system demonstrating the complete journey from **machine learning experimentation to a containerized and cloud-deployed application**.

> **Note:** This project is intended for educational and demonstration purposes and should not be used as a substitute for professional medical diagnosis.
