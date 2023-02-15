# ETL pipelines
- **ETL** is best used for on-premise data that needs to be structured before uploading it to a relational data warehouse. This method is typically implemented when datasets are small and there are clear metrics that a business values because large datasets will require more time to process and parameters are ideally defined before the transformation phase. This is a data warehouse solution.
- **ELT** is best suited for large volumes of data and implemented in cloud environments where the large storage and computing power available enables the data lake to quickly store and transform data as needed. ELT is also more flexible when it comes to the format of data but will require more time to process data for queries since that step only happens as needed versus ETL where the data is instantly queryable after loading. This is a data lake solution.
***

## ML Pipeline
- This concerns how to write pipeline for model deployment. 
- Compared to an ETL pipeline, it is still a pipeline, but it follows the whole process up to deployment. So a Ml pipeline incorporate an ETL pipeline.
***

## A note on the notebook rendering
Each notebook has two versions (all python scripts are unaffected by this):
- One where all the markdown comments are rendered in black& white. These are placed in the folder named `GitHub_MD_rendering` where MD stands for MarkDown.
- One where all the markdown comments are rendered in coloured.
***

## Available tutorials
- [ETL_in_Keras_TF](https://github.com/kyaiooiayk/ETL-Pipelines-Notes/blob/main/tutorials/GitHub_MD_rendering/ETL_in_Keras_TF.ipynb)
- [Simple ETL pipeline](https://github.com/kyaiooiayk/ETL-Pipelines-Notes/blob/main/tutorials/GitHub_MD_rendering/Simple%20ETL%20pipeline.ipynb)
- [From jupyter notebook to ML production pipeline with GitHub actions](https://github.com/kyaiooiayk/CI-CD-Pipeline-with-GitHub-Actions)
***

## References
- [Data Pipelines Explained - YouTube](https://www.youtube.com/watch?v=6kEGUCrBEU0)
- [ETL vs ELT - Article](https://blog.hubspot.com/website/etl-vs-elt)
- [How to Unit Test Deep Learning](https://theaisummer.com/unit-test-deep-learning/)
***
