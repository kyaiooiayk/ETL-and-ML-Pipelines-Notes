# ETL pipelines
- **ETL** is best used for on-premise data that needs to be structured before uploading it to a relational data warehouse. This method is typically implemented when datasets are small and there are clear metrics that a business values because large datasets will require more time to process and parameters are ideally defined before the transformation phase. This is a data warehouse solution.
- **ELT** is best suited for large volumes of data and implemented in cloud environments where the large storage and computing power available enables the data lake to quickly store and transform data as needed. ELT is also more flexible when it comes to the format of data but will require more time to process data for queries since that step only happens as needed versus ETL where the data is instantly queryable after loading. This is a data lake solution.
***

## ML Pipeline
- This concerns how to write pipeline for model deployment. 
- Compared to an ETL pipeline, it is still a pipeline, but it follows the whole process up to deployment. So a Ml pipeline incorporate an ETL pipeline.
***

## How to test a pipeline
- Testing Data Pipelines is hard for the following reasons:
  - At least in ML you do not expect them to produce the same output over and over again.
  - There is a source of pseudo randomness.
  - Because a pipeline process data, this is not the same thing as testing your source code.

### Make data assumptions explicit
- Once this function is created, created some unit tests to check if both exceptions are raied.
```python
def load_features(list_of_features):
    """Load features from the database.
    """
    uri = load_uri()
    con = db_connection(uri)
    tables = load_from_db(list_of_features, con)
    if 'outcome' in tables:
        raise Exception('You cannot load the outcome variable as a feature')
    if any(tables, has_nas):
        raise Exception('You cannot load features with NAs')
    return tables
```

### Test your code with a fake database
- A fast test is better than slow test, but a slow test is better than not testing at all.
- Use fake database to test your pipeline
```python
# db.py
import os

def db_connection():
    if os.environ['testing_mode']:
        return connect_to_testing_db()
    else:
        return connect_to_production_db()
```

```shell
# run_tests.sh
export testing_mode=YES
py.test # run your tests
export testing_mode=NO
```
### Dealing with randomness
- Unfortunately fixing the seed is much more difficult than this.
```python 
# test_random.py
# set the seed
set_random_seed(0)

def test_generate_random_number_below_10():
    assert generate_random_number_between(0, 10) == 5

def test_generate_random_number_between_8_12():
    assert generate_random_number_between(8, 12) == 10
```
- Another approach is to test your function enough number and check the result against an interval and not an specific value. Let's see how to test a function that draws one sample from the normal distribution:
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
- [How to test your ML pipeline](https://dssg.github.io/hitchhikers-guide/curriculum/programming_best_practices/test-test-test/ds_testing/)
***

## References
- [Data Pipelines Explained - YouTube](https://www.youtube.com/watch?v=6kEGUCrBEU0)
- [ETL vs ELT - Article](https://blog.hubspot.com/website/etl-vs-elt)
- [How to Unit Test Deep Learning](https://theaisummer.com/unit-test-deep-learning/)
***
