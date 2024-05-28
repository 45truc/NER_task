# NER_task
NLP Deliverable 2

Some details:

*  main.pdf is a document containing basic explanations of your work and your results.
In particular you need to explain how the structure perceptron works in detail. For
instance, explain what happens if a word that is never seen during training is presented
to the model at test time. This file should contain a brief section describing the work
carried out by each group member.
*  train models.ipynb is a notebook containing all the code required to train the models
and store them in fitted models
*  eproduce results.ipynb is a notebook that must: load the data, load the fitted models from disk and evaluate the models, following the rules for the deliverable described
in this document.
*  fitted models is a folder where you will save your trained models. Any model used
in reproduce results.ipynb should be imported from fitted models.
*  utils is a folder containing functions that might be used in the train models.ipynb
and/or reproduce results.ipynb. The goal is to keep the notebooks clean (avoid
writing large functions inside the notebook, if you do it you can take them out, write
them in utils.py, import them and use them).
