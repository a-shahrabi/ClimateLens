# 🌍 ClimateLens

ClimateLens is an open-source research project dedicated to understanding and addressing climate anxiety among young people. Our goal is to identify how climate anxiety manifests in youth and to develop a reproducible, reusable, and interpretable detection model for early intervention. By analyzing language patterns and expressions linked to climate anxiety, the project uncovers common themes and concerns, offering insights into how youth experience and cope with this growing issue.

## Problem Statement

Climate change is not only a global environmental challenge but also a significant source of anxiety. Despite its increasing prevalence, there is limited understanding of how climate anxiety manifests and few tools exist for early detection and intervention. Without timely support, these anxieties can escalate, worsening mental health outcomes and reducing overall well-being.

We hypothesize that:

+ By uncovering common themes, concerns, and expressions related to climate anxiety,
+ And by developing a robust NLP-based or LLM-based model to accurately identify instances of climate anxiety,

We can provide recommendations for effective interventions and support mechanisms. Ultimately, this will improve mental health outcomes and well-being for youth affected by climate anxiety.

The successful execution of this project will:
+ Provide invaluable insights into how climate anxiety manifests.
+ Enable the development of a practical detection and intervention tool.
+ Ensure scalability and accessibility by leveraging open-source datasets.
+ Offer an interactive tool for model implementation and insight visualization.

- [🌐 Launch Webapp](https://huggingface.co/spaces/crc-sprout/ClimateLens)  
- [📖 Learn More](https://crc.place/climatelens/)

## ✨ Features
- **Data Collection** – tools for gathering and cleaning social media datasets.  
- **NLP Models** – topic modeling and classification for detecting climate-related emotions.  
- **Visualization** – interactive graphics and dashboards.  
- **WebApp** – HuggingFace Space.  

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

## Impact
By enabling early identification of climate anxiety, this project empowers support networks and mental health professionals to respond proactively. It will also contribute valuable research on climate-related perspectives, helping shape better interventions and policies.

Our ultimate goals are to:
+ Build resilience in young people.
+ Provide a sense of agency.
+ Help transform climate-related fears into constructive engagement.

## 🌐 WebApp
The production app is deployed on HuggingFace Spaces using Streamlit. All visualizations and explanations are present in the app.

## 🤝 Contributing
This is an organization-only project.

## License
This project is licensed under the MIT License – see the LICENSE file for details.