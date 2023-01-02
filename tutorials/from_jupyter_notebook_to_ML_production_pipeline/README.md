# From jupyter notebook to ML production pipeline

- Following a nice 3-series article published on Medium:
    - [Back to the Machine Learning fundamentals: How to write Pipeline for Model deployment (Part 1/3)](https://ivannardini.medium.com/back-to-the-machine-learning-fundamentals-how-to-write-code-for-model-deployment-part-1-3-4b05deda1cd1)
    - [Back to the Machine Learning fundamentals: How to write Pipeline for Model deployment (Part 2/3)]()
    - [Back to the Machine Learning fundamentals: How to write Pipeline for Model deployment (Part 3/3)]()
 ***
 
 ## Step 0
 - Download the two notebooks published on Kaggle mentioned in the article. These will be considered the baseline and the code published under the ML productin pipeline is based on these two notebooks. I have downloaded and saved them in this repository ()
     - [Notebook #1 by OmkarReddy Kaggle user]()
     - [Notebook #2 by Bunty Shah Kaggle user]()
- When it comes to replicating the result, this is not possible because the two authors have not fix the seed the `train_test_split` sklearn function making their model training essentially random.
***

## Step #1 - Procedural pipeline
- The first rule while trying to create a model deployment pipeline is to always set the random seed in oder to make sure your cod e is reproducible.
- The rewritten codes can be found [here]() where there is a notebooks for each steps. 
- A typical Machine learning pipeline for production is mainly composed of four (give or take) steps:
    - Data Gathering
    - Features engineering
    - Features selection
    - Model Building
- In this cases these are slightly different:
    - 1.0-insurance-Data_Analysis.ipynb
    - 2.0-insurance-Feature_Engineering.ipynb
    - 3.0-insurance-Feature_Selection.ipynb
    - 4.0-insurance-Model-Building.ipynb
    - 5.0-insurance-Model-Deployment.ipynb
*** 


