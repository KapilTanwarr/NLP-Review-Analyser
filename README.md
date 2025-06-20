# 🧠 NLP Review Analysis Dashboard

A full-stack machine learning project that analyzes customer reviews using **Natural Language Processing (NLP)** techniques including **Sentiment Analysis** (VADER) and **Topic Modeling** (BERTopic), with results visualized in an interactive **Streamlit dashboard**.

---

## 📌 Features

- ✅ Clean and preprocess real-world review text  
- ✅ Perform sentiment classification: Positive, Negative, Neutral  
- ✅ Discover latent topics using BERTopic  
- ✅ Visualize key insights with Plotly charts  
- ✅ Fully interactive dashboard with Streamlit  
- ✅ Ready-to-run pipeline in `.py` and `.ipynb` formats

---

## 📂 Project Structure

nlp-review-analysis/
├── data/
│ └── reviews.csv # Dataset (input/output)
├── notebooks/
│ └── nlp_pipeline.ipynb # Jupyter Notebook (optional)
├── app/
│ └── streamlit_app.py # Streamlit Dashboard
├── nlp_pipeline.py # Main NLP pipeline script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file

# Install Required Packages
It’s recommended to use a virtual environment:
pip install -r requirements.txt
Run the NLP Pipeline
python nlp_pipeline.py
This will clean the data, run sentiment analysis & topic modeling, and update data/reviews.csv.
Launch the Streamlit Dashboard
streamlit run app/streamlit_app.py
Then open the browser (if it doesn't open automatically) and go to:
rduino
http://localhost:8501
📊 Dashboard Preview

The dashboard includes:

📈 Sentiment Distribution Chart (Positive/Neutral/Negative)

🧠 Topic Frequency Bar Chart using BERTopic

📋 Interactive Table of all reviews with sentiment and topic annotations

# 💡 Tech Stack
Python · Pandas · Regex · NLTK

VADER Sentiment Analyzer

BERTopic for unsupervised topic modeling

Plotly for charts

Streamlit for web app UI

# 📁 Dataset
This project uses customer reviews from the Amazon Reviews Dataset. You can replace C:\Python\data\reviews.csv with your own dataset (just ensure there's a Text column or rename it in the code).

# 🔄 Customization Ideas
Integrate HuggingFace transformers for advanced classification

Add word clouds or TF-IDF visualizations

Deploy on Streamlit Cloud, Heroku, or Render

Use LDA as an alternative to BERTopic

Train sentiment classifier using labeled data

🤝 Contribution
Feel free to fork this repo, make improvements, and submit a pull request!
Got ideas to improve the pipeline or dashboard? Let's collaborate.

🏷️ Tags
nlp sentiment-analysis topic-modeling bertopic vader streamlit python data-visualization machine-learning
