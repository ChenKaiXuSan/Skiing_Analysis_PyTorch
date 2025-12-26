<div align="center">    
 
# Skiing Motion Analysis

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->

<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

<!--
Conference
-->
</div>
 
## Description   
This repository is a research-oriented deep learning framework for skiing motion analysis, focusing on video-based 3D pose estimation, multi-view reconstruction, and biomechanical performance evaluation.

The project is built on PyTorch Lightning and designed for sports engineering and computer vision research, enabling scalable experimentation on complex skiing movements captured from monocular or multi-camera videos.

Our goal is to quantitatively analyze skiing techniques by modeling:

- Full-body 3D kinematics
- Temporal motion patterns
- Inter-joint coordination and stability
- Athlete-specific movement characteristics

## How to run

First, install dependencies

```bash
# clone project
git clone [url]

# install project
cd deep-learning-project-template
pip install -e .
pip install -r requirements.txt
```

Next, navigate to any file and run it.

```bash
# module folder
cd project

# run module (example: mnist as your main contribution)
python lit_classifier_main.py
```

## Project Organization

```txt
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── project            <- Source code for use in this project.
│   ├── __init__.py    <- Makes project a Python module
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
