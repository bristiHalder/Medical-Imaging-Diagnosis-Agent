# 🏥 Medical Imaging Diagnosis Agent

A **Medical Imaging Diagnosis Agent** built using **Agno Agents** and **Google Gemini 2.5 Pro**, providing **AI-assisted analysis of medical images** such as X-rays, CT scans, MRIs, ultrasounds, and DICOM files.

This application leverages **multimodal large language models** to perform structured radiological analysis, highlight potential abnormalities, and explain findings in both **clinical** and **patient-friendly** language.


---

## 🚀 Features

### 🔍 Comprehensive Medical Image Analysis
- Automatic **image modality identification** (X-ray, MRI, CT, Ultrasound, etc.)
- **Anatomical region detection** and patient positioning
- **Image quality and technical adequacy assessment**

### 🧠 AI-Assisted Radiological Insights
- Systematic **key findings and observations**
- Detection of **potential abnormalities**
- Severity estimation (Normal / Mild / Moderate / Severe)
- AI-assisted **radiological impressions** (non-definitive)

### 🧑‍⚕️ Patient-Friendly Explanation
- Simplified explanations using clear language
- Medical jargon reduction with easy definitions
- Visual analogies where helpful
- Addresses common patient concerns

### 📚 Research & Reference Integration
- Automated medical literature lookup using **DuckDuckGo**
- References to recent studies and standard protocols
- Links to relevant medical resources and technologies

### 🖼️ Multi-Format Image Support
- JPG / JPEG / PNG
- DICOM (.dcm) medical images

### 🔐 Secure API Key Handling
- Google API key input via Streamlit sidebar
- Session-based key storage
- Easy reset and reconfiguration

---

## 🧱 Tech Stack

- **Frontend**: Streamlit  
- **AI Agent Framework**: Agno  
- **LLM**: Google Gemini 2.5 Pro  
- **Medical Imaging**: Pillow, pydicom, OpenCV  
- **Search Tool**: DuckDuckGo  
- **Language**: Python  

---



