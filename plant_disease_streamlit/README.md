# Plant Leaf Disease Detection — Streamlit

## Files
- `app.py`: Streamlit user interface and image prediction.
- `train_model.py`: trains the CNN on TensorFlow Datasets' PlantVillage dataset.
- `requirements.txt`: Python dependencies.

## 1. Train the model
Run this locally from this folder (training downloads PlantVillage on its first run):

```bash
python -m pip install -r requirements.txt
python train_model.py
```

Training creates:
- `plant_leaf_disease_cnn.keras`
- `class_names.json`

The training settings and CNN architecture follow the supplied notebook. The notebook's training split is retained; the script uses its training and validation portions for fitting.

## 2. Test the app locally
Keep `app.py`, `plant_leaf_disease_cnn.keras`, and `class_names.json` in the same folder, then run:

```bash
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud
Push all four required files (`app.py`, `requirements.txt`, `plant_leaf_disease_cnn.keras`, and `class_names.json`) to your GitHub repository. In Streamlit Community Cloud, choose the repository and set **Main file path** to:

```text
app.py
```

Note: the trained `.keras` model is a required deployment artifact. Do not deploy only the notebook and expect the app to predict without it. Check the model file size and GitHub's file-size limits before pushing; Git LFS or external model storage may be needed for a large model.
