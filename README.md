<div align="center">
<h1>Ictal-Phase Detection</h1>
<h3>Enhanced Non-EEG Multimodal Seizure Detection: A Real-World Model for Identifying Generalised Seizures Across The Ictal State</h3>

[Jamie Pordoy](https://github.com/Unrealluver)<sup>1</sup> \*,[Graham Jones](https://github.com/LegendBC)<sup>2</sup> \*,[Nasser Mattorianpour](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>1</sup>, ,[Molly Evans](https://github.com/LegendBC)<sup>3</sup> \*, [Nasim Dadeshi](https://www.xloong.wang/)<sup>1</sup>, [Massoud Zolgharni](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>

<sup>1</sup>University of West London<sup>2</sup>Open Seizure Detector<sup>3</sup> Pleotek LTD

Corresponding Author: Dr. Jamie Pordoy <sup>jamiepordoy@hotmail.com</sup>

TechrXiv Preprint ([TechrXiv 1095532](https://www.techrxiv.org/users/692829/articles/1095532-enhanced-non-eeg-multimodal-seizure-detection-a-real-world-model-for-identifying-generalised-seizures-across-the-ictal-state)), AMBER Model (https://github.com/jpordoy/AMBER)


</div>


#

[Latest Update] News

* **September 5th, 2024:** We have created a single Python file called myScript.py that contains the full codebase, making it easier to adapt and run in a Colab or IPYNB environment.

* **July 21st, 2024:**  The AMBER model has been made open source, and we are beginning to prepare version 1. Please get in touch! Further details can be found in the code and our TechArXiv preprint.

* **June 12, 2024:** We are pleased to announce the release of our paper on TechRxiv. The accompanying code and models will be available soon. Stay tuned for updates! ☕️


## Abstract
Non-electroencephalogram seizure detection models hold promise for the early detection of generalised onset seizures. However, these models often experience high false alarm rates and difficulties in distinguishing normal movements from seizure manifestations. To address this, we were granted exclusive access to the newly developed Open Seizure Database, from which a representative dataset of 94 events was selected (42 generalised tonic-clonic seizures, 19 auras/focal seizures, and 33 seizures labelled as Other), with a combined duration of approximately 5 hours and 29 minutes. Each event contains acceleration and heart rate data which was expertly annotated by a clinician, who labelled every 5 second timestep with a class of Normal, Pre-Ictal, or Ictal. We then introduced the AMBER (Attention-guided Multi-Branching-pipeline with Enhanced Residual fusion) model. AMBER constructs multiple branches to form independent feature extraction pipelines for each sensing modality. The outputs of each branch are passed to our custom Enhanced Residual Fusion layer, where the extracted features are combined into a fused representation. The fused representation is then propagated through two densely connected blocks before being passed through a softmax activation function. The model was trained using k-fold cross validation, with k-1 fold used to train the model and the remaining fold was used to evaluate the model’s performance. The results of these experiments underscore the efficacy of Ictal-Phase Detection, achieving an accuracy and f1-score of 0.8995 and 0.8987. Notably, the model exhibited consistent generalisation, recording a True Positive Rate of 0.9564, 0.8325, and 0.9111 for the Normal, Pre-Ictal, and Ictal classes respectively. These findings were compounded by an average False Positive Rate, recording an overall score of 0.0502. In conclusion, this research introduces a new detection technique and model designed for multimodal seizure detection, with the potential to reduce the false alarm window and differentiate high and low amplitude convulsive movement. We believe the results of this study lay the groundwork for further advancements in non-electroencephalogram seizure detection research.
<br />

<div align="center">
<img src="https://github.com/jpordoy/Ictal-Phase-Detection/blob/main/Manuscript_Images/model.png" />
</div>

## Envs. for Pretraining

- Python 3.7.17

  - `conda create -n your_env_name python=3.7.17`

- Tensorflow version 2.15.0
  - `pip install tensorflow==2.15.0  --index-url (https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0-rc1)`


<!-- Getting Started -->

Clone the project

```bash
  git clone https://github.com/jpordoy/Ictal-Phase-Detection.git
```

Go to the project directory

```bash
  cd Ictal-Phase-Detection
```

Install dependencies

```bash
  pip install -r requirements.txt
```

### How To Run The Code
Please put your training data as a csv file in the "Data/" of this project.

```python        
import pandas as pd
import numpy as np
from data_loader import DataLoader
from data_formatter import DataFormatter
from model import Amber
from kfold_cv import KFoldCrossValidation
from evaluator import evaluate_model_performance
from config import config

# Define your DataFrame and parameter
mypath = 'Data/Train.csv'
df = pd.read_csv(mypath)
target_column = 'label'  # Name of the target column

# Step 1: Load Data
data_loader = DataLoader(dataframe=df, time_steps=config.N_TIME_STEPS, step=config.step, target_column=target_column)
segments, labels = data_loader.load_data()

# Step 2: Format Data
data_formatter = DataFormatter(config=config)
X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(segments, labels)

# Reshape y_test correctly
y_test_reshaped = np.asarray(y_test, dtype=np.float32)

# Initialize model
ts_model = Amber(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=config.N_CLASSES)

# Create an instance of KFoldCrossValidation
kfold_cv = KFoldCrossValidation(ts_model, [X_train_reshaped['Feature_1'], X_train_reshaped['Feature_2']], y_train)

# Run the cross-validation
kfold_cv.run()

# Evaluate the model performance
evaluation_results = evaluate_model_performance(ts_model, [X_test_reshaped['Feature_1'], X_test_reshaped['Feature_2']], y_test_reshaped)

# Access individual metrics
print("Accuracy:", evaluation_results["accuracy"])
print("F1 Score:", evaluation_results["f1"])
print("Cohen's Kappa:", evaluation_results["cohen_kappa"])

```




<!-- Contributing -->
## :wave: Contributing

<a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" />
</a>


Contributions are always welcome!

Please email at jamiepordoy@hotmail.com for ways to get started.

<!-- Code of Conduct -->
### :scroll: Code of Conduct

## Acknowledgement:
We extend our sincere gratitude to Open Seizure Detector for
providing early access to the Open Seizure Database. Their steadfast
dedication to open research will greatly enhance the progress of non-
EEG seizure detection research. We wholeheartedly acknowledge and
commend their unwavering support and commitment to advancing
scientific exploration and knowledge within the epilepsy community.
Additionally, we express our appreciation to the participants of the
Open Seizure Database. Without their dedication and commitment,
this research would not be feasible.

## AVAILABILITY OF DATA AND MATERIALS:
We are committed to advancing non-EEG detection research and
are pleased to announce the availability of our study’s results, models,
and code base. The AMBER model is now distributed under the MIT
license as open-source [41]. We invite any technical contributions
aimed at enhancing the model’s performance and extending its
application in non-EEG seizure detection. The full code base and
results for this project are now accessible on GitHub

## Citation
If you find this code useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
 @article{IPD24,
  title={Enhanced Non-EEG Multimodal Seizure Detection: A Real-World Model for Identifying Generalised Seizures Across The Ictal State},
  author={J. Pordoy, G. Jones, N. Matoorian, et al},
  journal={TechRxiv preprint},
  year={2024}
  doi={DOI: 10.36227/techrxiv.171822381.12552111/v1}
}
```
