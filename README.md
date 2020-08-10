# Invoice_Processing

- An invoicing tool that uses image processing to extract text bounding boxes, text contained and classification of text into field or detail.

## Installation :

```
git clone https://github.com/max-lulz/Invoice_Processing.git
cd into the repo directory
pip install -r requirements.txt
pip uninstall torch
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html  
```

- You will need Tesseract installed on your system
  - Sudo apt install tesseract-ocr
  
# Running :
 
``` 
python FieldClassifier.py
```

