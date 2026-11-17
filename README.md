# Smart Traffic Congestion Predictor
Predicts a Low/Medium/High congestion category and estimates average speed from time, traffic volume, rain, visibility, and road type.

Dataset: 1,800 synthetic rows in `dataset/traffic_training_data.csv`.
Training: two Random Forest pipelines; models are saved in `models/`.
Run:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train_model.py
python src/predict.py
streamlit run app.py
```
Regenerate CSV: `python src/generate_dataset.py`. Synthetic data is for education only, not operational traffic or emergency decisions.
