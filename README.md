```bash
conda install -c conda-forge \
    pytorch=2.2.2 \
    faiss-cpu \
    spacy=3.7.2 \
    numpy=1.26.4 \
    pandas=2.2.2 \
    scikit-learn=1.4.2 \
    tqdm=4.66.2 \
    nltk=3.8.1 \
    pip -y

pip install \
    transformers==4.40.2 \
    accelerate==0.29.2 \
    sentence-transformers==2.7.0 \
    datasets==2.19.1 \
    lemminflect==0.2.3 

pip install nltk
pip install spacy
python -m spacy download en_core_web_sm
```