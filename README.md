To run the main file (Group 27_COMP90042...), you need to follow the instructions to establish a python environment with some libraries of some certain version and activate it. 
```bash
conda env create -f environment.yml
conda activate nlp_final
python -m ipykernel install --user --name nlp_final --display-name "Python (nlp_final)"
```

Moreover, you need to download the spacy model for English language. The following commands can help you to do that.

```bash
python -m spacy download en_core_web_sm
```

Finally, differentiate the file path if you run the code on your local machine or on Google colab.
```bash
# for colab
from google.colab import drive
drive.mount('/content/drive')
train_claims_path = '/content/drive/MyDrive/data/train-claims.json'
dev_claims_path = '/content/drive/MyDrive/data/dev-claims.json'
evidence_path = '/content/drive/MyDrive/data/evidence.json'

# for local machine
train_claims_path = './data/train-claims.json'
dev_claims_path = './data/dev-claims.json'
evidence_path = './data/evidence.json'
```

In the process of executing the code, there will be several save points, the successfully trained models will be saved to a pre-defined directory and get loaded when needed. You're not required to re-train the model every time, you can just execute the code section which is the definition of `load_model` and `load_tokenizer` to load the pre-trained models.