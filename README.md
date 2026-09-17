# 🧠 Brain Tumor AI — Full-Stack MRI Classification & Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/AWS-EC2-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS EC2"/>
</p>

<p align="center">
  <strong>A complete end-to-end AI application for brain MRI classification, explainable predictions, authentication, containerization, and cloud deployment.</strong>
</p>

---

## 📌 Overview

A production-ready, full-stack AI platform for **multi-class brain tumor classification from MRI scans**.

The system combines:

* 🧠 **TensorFlow/Keras CNN** for MRI classification
* 🔍 **LIME Explainable AI (XAI)** for model interpretability
* ⚡ **FastAPI REST backend**
* 🎨 **Nginx-served frontend**
* 🔐 **SQLAlchemy-based authentication**
* 🗄️ **Persistent SQLite database**
* 🐳 **Docker & Docker Compose**
* 📦 **Docker Hub image distribution**
* ☁️ **AWS EC2 cloud deployment**

> ⚠️ **Medical Disclaimer:** This project is intended strictly for research, educational, and portfolio demonstration purposes. It is **not a certified clinical diagnostic tool**.

---

## 💡 Quick Overview

| 🏷️ Domain               | 🚀 Key Highlights                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------- |
| 🧠 **Machine Learning**  | Custom CNN, data augmentation, multi-class Softmax classification                                   |
| 🔍 **Explainable AI**    | LIME-based region-of-interest heatmaps                                                              |
| ⚡ **Backend & Database** | FastAPI REST API, OpenAPI/Swagger, SQLAlchemy ORM, SQLite                                           |
| 🎨 **Frontend UI**       | Responsive Vanilla JS dashboard, real-time image preview, probability visualization, authentication |
| 🐳 **DevOps & Cloud**    | Docker multi-container architecture, Docker Hub, AWS EC2                                            |

### 🎯 Supported Classes

```text
🧠 Glioma
🧠 Meningioma
🧠 Pituitary
🟢 No Tumor
```

---

# 🏗️ System Architecture

The application follows a **frontend → API → AI inference → database/explainability** architecture.

```text
                         🌐 CLIENT
                            │
                            ▼
              ┌─────────────────────────────┐
              │       👤 User Browser       │
              │                             │
              │   HTML5 / CSS3 / JavaScript │
              └──────────────┬──────────────┘
                             │
                    HTTP :8080
                             │
                             ▼
              ┌─────────────────────────────┐
              │     🎨 FRONTEND CONTAINER  │
              │                             │
              │          Nginx Alpine       │
              └──────────────┬──────────────┘
                             │
                    REST API :8000
                             │
                             ▼
              ┌─────────────────────────────┐
              │     ⚡ BACKEND CONTAINER    │
              │                             │
              │           FastAPI           │
              │                             │
              │   ┌─────────────────────┐   │
              │   │ 🧠 TensorFlow CNN   │   │
              │   ├─────────────────────┤   │
              │   │ 🔍 LIME Engine      │   │
              │   └─────────────────────┘   │
              │              │              │
              │   SQLAlchemy + SQLite       │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ☁️ AWS EC2 Instance
```

---

# 🔄 Application Workflow

The application follows an end-to-end workflow from authentication to MRI classification and explainable prediction.

```text
┌─────────────────────────┐
│        👤 USER          │
│     Opens Web App       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    🔐 REGISTER / LOGIN  │
│     Authentication      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    🧠 UPLOAD MRI IMAGE  │
│      Preview Scan       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   🎨 FRONTEND JAVASCRIPT│
│     Sends API Request   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     ⚡ FASTAPI BACKEND  │
│        /predict         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   🛠️ IMAGE PROCESSING  │
│    Resize + Normalize   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    🧠 TENSORFLOW CNN    │
│       Model Inference   │
└────────────┬────────────┘
             │
             ▼
┌──────────────────────────────┐
│      📊 CLASSIFICATION        │
│                              │
│  🧠 Glioma                   │
│  🧠 Meningioma               │
│  🧠 No Tumor                 │
│  🧠 Pituitary                │
└──────────────┬───────────────┘
               │
               ├──────────────────────┐
               │                      │
               ▼                      ▼
     ┌───────────────────┐   ┌───────────────────┐
     │ 📊 CLASS          │   │ 🔍 LIME           │
     │ PROBABILITIES     │   │ ANALYSIS          │
     │ Visualization     │   │ Explain Prediction│
     └─────────┬─────────┘   └─────────┬─────────┘
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                ┌────────────────────────┐
                │    🎯 RESULTS DASHBOARD│
                │                        │
                │  • Prediction          │
                │  • Probabilities       │
                │  • LIME Explanation    │
                └────────────────────────┘
```

---

# 🧠 Machine Learning Pipeline

The application uses a **TensorFlow/Keras CNN** to classify brain MRI scans into four categories:

**Glioma · Meningioma · Pituitary · No Tumor**

```text
           🧠 MRI Image
                │
                ▼
       📐 Resize 224×224
                │
                ▼
       🔢 Normalize 0–1
                │
                ▼
       🔄 Data Augmentation
                │
                ▼
          🧠 CNN Model
                │
                ▼
      📊 Softmax Probabilities
                │
                ▼
        🎯 Predicted Class
```

---

# 🔍 Explainable AI — LIME

To improve model interpretability, the application integrates **LIME (Local Interpretable Model-Agnostic Explanations)**.

After generating a prediction, the system analyzes the MRI image and identifies regions that contributed most to the model's decision.

```text
        🧠 MRI Image
              │
              ▼
       🧠 CNN Prediction
              │
              ▼
        🔍 LIME Analysis
              │
              ▼
    🟥 Important Image Regions
              │
              ▼
       💡 Visual Explanation
```

### ✨ Explainability Flow

```text
Prediction
    │
    ▼
LIME Perturbation
    │
    ▼
Model Responses
    │
    ▼
Important Regions
    │
    ▼
Heatmap / Explanation
```

---

# ⚡ FastAPI Backend

The application uses **FastAPI** to provide a lightweight and scalable REST API for authentication, MRI prediction, and explainable AI.

## 🔌 API Endpoints

| Method | Endpoint    | Description                              |
| :----: | :---------- | :--------------------------------------- |
|  `GET` | `/`         | 🩺 API health/status check               |
| `POST` | `/register` | 👤 Register a new user                   |
| `POST` | `/login`    | 🔐 Authenticate an existing user         |
| `POST` | `/predict`  | 🧠 Classify a brain MRI image            |
| `POST` | `/explain`  | 🔍 Generate a LIME explanation           |
|  `GET` | `/docs`     | 📚 Interactive Swagger API documentation |

## 🛠️ Backend Responsibilities

* 👤 User registration and authentication
* 📤 MRI image upload handling
* 🧠 Brain tumor classification using the trained CNN
* 📊 Prediction probability generation
* 🔍 LIME-based visual explanations
* 🗄️ SQLite database integration using SQLAlchemy
* 🔌 REST API communication with the frontend

FastAPI's built-in **Swagger UI** makes it easy to test and explore the available endpoints during development and deployment.

---

# 🔐 Database & Authentication

The application includes a user authentication system backed by **SQLite** and **SQLAlchemy**.

## 🔄 Authentication Flow

### 👤 Registration

```text
👤 User Registration
        │
        ▼
⚡ FastAPI API
        │
        ▼
🗄️ SQLite Database
        │
        ▼
✅ User Account Created
```

### 🔑 Login

```text
👤 User Login
      │
      ▼
⚡ FastAPI Authentication
      │
      ▼
🔐 Validate Credentials
      │
      ▼
🎯 Access Dashboard
```

---

# 🎨 Frontend

The frontend provides the user-facing interface for authentication, MRI image upload, prediction results, and explainable AI visualization.

## ✨ Features

* 👤 User registration and login
* 🧠 Brain MRI image upload
* ⚡ Real-time communication with the FastAPI backend
* 📊 Predicted tumor class and probability display
* 🔍 LIME explanation visualization
* 🎯 Dashboard-based user experience
* 📱 Responsive interface using HTML and CSS

## 💻 Technology

| Technology     | Purpose                      |
| :------------- | :--------------------------- |
| **HTML5**      | 🧱 Page structure            |
| **CSS3**       | 🎨 Styling and responsive UI |
| **JavaScript** | ⚡ Client-side functionality  |
| **Nginx**      | 🌐 Frontend web server       |

The frontend dynamically determines the backend API host from the current browser hostname, allowing the same frontend build to work across local Docker and AWS EC2 deployments.

---

# 🐳 Docker Architecture

The application is containerized using **Docker**, with separate containers for the backend API and frontend web server.

## 📦 Container Architecture

```text
                         👤 USER
                           │
                           ▼
              ┌──────────────────────────┐
              │     🌐 WEB BROWSER       │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  🎨 FRONTEND CONTAINER   │
              │                          │
              │  Nginx + HTML/CSS/JS     │
              │       Port: 8080         │
              └────────────┬─────────────┘
                           │
                   HTTP API Requests
                           │
                           ▼
              ┌──────────────────────────┐
              │   ⚡ BACKEND CONTAINER   │
              │                          │
              │  FastAPI + TensorFlow    │
              │       Port: 8000         │
              └────────────┬─────────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
        ┌─────────────────┐  ┌──────────────────┐
        │ 🗄️ SQLite DB    │  │ 🧠 CNN Model     │
        │ brain_tumor.db  │  │ best_model.keras │
        └─────────────────┘  └──────────────────┘
```

---

# 📦 Docker Hub

The containerized application is published to **Docker Hub** so the deployment environment can pull pre-built images without rebuilding the application on the server.

## 🔄 Image Workflow

```text
💻 Source Code
      │
      ▼
🐳 Build Docker Images
      │
      ▼
📦 Push Images to Docker Hub
      │
      ▼
☁️ AWS EC2
      │
      ▼
⬇️ Pull Images
      │
      ▼
🚀 Run Containers
```

---

# ☁️ AWS EC2 Deployment

The application is deployed on **Amazon EC2** using Docker containers.

## 🖥️ Deployment Environment

| Configuration           | Value                 |
| :---------------------- | :-------------------- |
| ☁️ **Cloud Provider**   | AWS                   |
| 🖥️ **Service**         | Amazon EC2            |
| 🐧 **Operating System** | Ubuntu Server         |
| ⚙️ **Instance Type**    | `t3.small`            |
| 🏗️ **Architecture**    | x86_64                |
| 💾 **Storage**          | 20 GiB gp3            |
| 🌏 **Region**           | Asia Pacific (Mumbai) |

## 🏗️ Deployment Architecture

```text
                    ☁️ AWS EC2 INSTANCE
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │ ⚡ BACKEND       │       │ 🎨 FRONTEND     │
      │ Docker Container │       │ Docker Container │
      │    Port 8000     │       │    Port 8080     │
      └────────┬────────┘       └─────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
 ┌──────────────┐ ┌──────────────┐
 │ 🗄️ SQLite DB │ │ 🧠 CNN Model │
 └──────────────┘ └──────────────┘
```

---

# 📈 Project Evolution

The project evolved from a standalone CNN-based classification model into a complete full-stack AI application.

## 🔄 Evolution Timeline

```text
🧠 CNN Model Development
          │
          ▼
🩻 Brain MRI Classification
          │
          ▼
📊 Model Evaluation
          │
          ▼
⚡ FastAPI Inference Backend
          │
          ▼
🔍 LIME Explainability
          │
          ▼
🔐 User Authentication + Database
          │
          ▼
🎨 Web-Based Frontend
          │
          ▼
🐳 Docker Containerization
          │
          ▼
📦 Docker Hub Image Distribution
          │
          ▼
☁️ AWS EC2 Deployment
```

---

# 🛠️ Engineering Highlights

This project demonstrates practical **software engineering, machine learning, and cloud deployment skills** alongside the core AI model.

## 🔧 Key Engineering Practices

| Area                       | Implementation                                          |
| :------------------------- | :------------------------------------------------------ |
| ⚡ **Backend**              | RESTful API using FastAPI                               |
| 🧠 **AI Inference**        | TensorFlow/Keras CNN                                    |
| 🔍 **Explainability**      | LIME-based visual explanations                          |
| 🗄️ **Database**           | SQLite + SQLAlchemy                                     |
| 🎨 **Frontend**            | Separate frontend/backend architecture                  |
| 🐳 **Containerization**    | Docker                                                  |
| 🔄 **Orchestration**       | Docker Compose                                          |
| 📦 **Distribution**        | Docker Hub                                              |
| ☁️ **Deployment**          | AWS EC2                                                 |
| 🔐 **Networking**          | AWS Security Groups                                     |
| 💾 **Persistence**         | SQLite database persisted outside backend container     |
| 🛡️ **Repository Hygiene** | `.gitignore` and `.dockerignore`                        |
| 🔬 **ML Integration**      | Original ML work extended into a deployable application |

---

# 🎯 Project Purpose

This project was developed to explore how a **machine learning model can be transformed into a complete, deployable AI application**.

Rather than stopping at model training and evaluation, the project combines:

| Component                   | Technology                     |
| :-------------------------- | :----------------------------- |
| 🧠 **Machine Learning**     | CNN-based MRI classification   |
| 🔍 **Explainable AI**       | LIME-based visual explanations |
| ⚡ **Backend Development**   | FastAPI REST API               |
| 🎨 **Frontend Development** | HTML, CSS, JavaScript          |
| 🔐 **Authentication**       | User registration and login    |
| 🗄️ **Database**            | SQLite with SQLAlchemy         |
| 🐳 **Containerization**     | Docker & Docker Compose        |
| 📦 **Image Distribution**   | Docker Hub                     |
| ☁️ **Cloud Deployment**     | AWS EC2                        |

---

## 🚀 End-to-End Journey

```text
          🧪 MACHINE LEARNING
                  │
                  ▼
          🧠 CNN DEVELOPMENT
                  │
                  ▼
          🩻 MRI CLASSIFICATION
                  │
                  ▼
          🔍 XAI / LIME
                  │
                  ▼
          ⚡ FASTAPI BACKEND
                  │
                  ▼
          🎨 WEB FRONTEND
                  │
                  ▼
          🔐 AUTHENTICATION
                  │
                  ▼
          🗄️ DATABASE
                  │
                  ▼
          🐳 DOCKERIZATION
                  │
                  ▼
          📦 DOCKER HUB
                  │
                  ▼
          ☁️ AWS EC2
                  │
                  ▼
          🚀 DEPLOYED AI APP
```

The result is an **end-to-end system** demonstrating the complete journey from **machine learning experimentation to a containerized and cloud-deployed application**.

---

> ⚠️ **Note:** This project is intended for educational and demonstration purposes and should not be used as a substitute for professional medical diagnosis.
