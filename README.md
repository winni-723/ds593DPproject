# Implementing Differential Privacy on RateMyProfessor Data  

## Abstract  
Online review platforms such as **RateMyProfessor** provide valuable insights for students choosing instructors — but they also expose sensitive data that may lead to **privacy violations**.  

This project builds a **prototype website** modeled after RateMyProfessor that integrates **Differential Privacy (DP)** for aggregated statistics. Using an **open RateMyProfessor dataset** as the backend, the system enables users to submit:  
- Ratings   
- Useful 
- Difficulty
- Would Take Again (Yes/No)
- Feedback 

All **aggregated outputs** will be released under **differential privacy guarantees**, while an **LLM-based text filter** will flag potentially identifiable information before submission.  

Based on preliminary experiments with ChatGPT, the model effectively detected overly specific or repeated identifiers that could leak information. Supporting screenshots will be provided to illustrate this capability.  

This project demonstrates how **modern privacy-preserving techniques** can be applied in **real-world systems** to balance **utility and privacy**.  

---

## 1. Plan of Work  

### 📊 Dataset Preparation  
- Load the open **RateMyProfessor dataset** from [Mendeley Data](https://data.mendeley.com/datasets/fvtfjyvw7d/2).  
- Remove personally identifiable information (PII).  
- Bound numeric/categorical attributes for DP application.  

### 💻 Platform Development  
- Build a prototype web platform using **Flask**, **Django**, or **FastAPI** with an SQL-based database.  
- Support user input:  
  - Rating (numerical)  
  - Difficulty (numerical)
  - Useful (numerical)
  - “Would Take Again” (boolean type)
  - Feedback (text)

### 🔐 Differential Privacy Integration  
- Apply **Laplace** or **Gaussian noise** to aggregated statistics (e.g., averages, proportions).  
- Display only **DP-protected** results on the site.  

### 🤖 LLM Text Filter  
- Detect identifiable or sensitive text by showing **high or low risk**  
- If detected, prompt the user with:  
  > “Your message may contain identifiable information. Do you want to rephrase it?”  

---

##  2. Techniques and Methods  

| Component | Tools / Frameworks |
|------------|--------------------|
| **Backend & Web Development** | Django / FastAPI, SQL Database |
| **Differential Privacy** | [Google PyDP](https://github.com/OpenMined/PyDP) |
| **LLM Filtering** | Gemini API |
| **Chatbot Integration** | Gemini |
| **Evaluation & Visualization** | Python, Matplotlib, Seaborn |

### 📂 Dataset  
We will use the **RateMyProfessor Dataset** from **Mendeley Data**, containing millions of student evaluations, including:  
- Overall quality ratings  
- Difficulty levels  
- “Would Take Again” 
- Useful
- written comments  

This dataset enables exploration of **student perspectives** on teaching effectiveness and course experience while testing **privacy-preserving mechanisms**.  

---

## Evaluation  

### **Structured Data (Ratings, Difficulty, Boolean)**  
- Compare **DP vs. non-DP** statistics using:  
  - Showing the **empirical distribution** of the differenical privacy noisy average  

### **Text Data (Feedback Comments)**  
- Test LLM filter on curated comments with identifiers.  
- Measure:  
  - **Precision** – Correctly flagged sensitive information  

---

##  3. Expected Outcomes  
This project will deliver a **functional prototype** demonstrating how **Differential Privacy** can protect user data in online review platforms like RateMyProfessor.  

We aim to show that:  
- Aggregated statistics can be released with **strong privacy guarantees** and **acceptable averages**.  
- Sensitive details in user feedback can be automatically detected and mitigated.  

This work extends beyond theoretical study by applying **DP and LLM filtering** to a **real-world dataset**, bridging the gap between **research and practice**.  


