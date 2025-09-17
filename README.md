# 🌍 ClimateLens

ClimateLens is a collaborative project by **Climate Resilient Communities (CRC)** that analyzes **climate anxiety in social media posts**.  
The project combines data science, machine learning, and web technologies to uncover climate-related themes and make insights accessible through an interactive webapp.

- [🌐 Launch Webapp](https://huggingface.co/spaces/crc-sprout/ClimateLens)  
- [📖 Learn More](https://crc.place/climatelens/)

## ✨ Features
- **Data Collection** – tools for gathering and cleaning social media datasets.  
- **NLP Models** – topic modeling and classification for detecting climate-related emotions.  
- **Visualization** – interactive graphics and dashboards.  
- **WebApp** – Hugging Face Space powered by FastAPI (backend) + Next.js (frontend).  

## 🔐 Required Environment Variables
```
# Cohere
COHERE_API_KEY=your_cohere_key
```

## 📂 Project Structure
```
├── data/                 # sample datasets
│
├── src/
│   ├── LDA/              # LDA model (baseline only)
│   ├── models/           # pipelines and ML models
│   ├── utils/            # shared utilities
│   └── notebooks/        # exploratory/archived notebooks
│
├── LICENSE
├── Makefile
├── requirements.txt
└── README.md
```

## 📖 Documentation
+ Data Schema
+ Starter Guide

## 🌐 WebApp
The production app is deployed on Hugging Face Spaces.

## 🤝 Contributing
This is an organization-only project.

## License
MIT License © 2025 Climate Resilient Communities