```bash
conda env create -f environment.yml
conda activate nlp_final
python -m ipykernel install --user --name nlp_final --display-name "Python (nlp_final)"
python -m spacy download en_core_web_sm
```