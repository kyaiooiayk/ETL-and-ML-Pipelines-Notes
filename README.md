# ETL pipelines
- **ETL** is best used for on-premise data that needs to be structured before uploading it to a relational data warehouse. This method is typically implemented when datasets are small and there are clear metrics that a business values because large datasets will require more time to process and parameters are ideally defined before the transformation phase.

- **ELT** is best suited for large volumes of data and implemented in cloud environments where the large storage and computing power available enables the data lake to quickly store and transform data as needed. ELT is also more flexible when it comes to the format of data but will require more time to process data for queries since that step only happens as needed versus ETL where the data is instantly queryable after loading.

***

## Available tutorials
- [ETL_in_Keras_TF](https://github.com/kyaiooiayk/ETL-Pipelines-Notes/blob/main/tutorials/GitHub_MD_rendering/ETL_in_Keras_TF.ipynb)
- [Simple ETL pipeline](https://github.com/kyaiooiayk/ETL-Pipelines-Notes/blob/main/tutorials/GitHub_MD_rendering/Simple%20ETL%20pipeline.ipynb)
***

## References
- [Data Pipelines Explained - YouTube](https://www.youtube.com/watch?v=6kEGUCrBEU0)
- [ETL vs ELT - Article](https://blog.hubspot.com/website/etl-vs-elt)
***