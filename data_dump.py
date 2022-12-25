import pymongo
import pandas as pd
import json
# Provide mongodb localhost uel to connect python to mongodb
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATABASE_NAME = 'aci'
COLLECTION_NAME = 'census'

if __name__ == '__main__':
    df = pd.read_csv("/config/workspace/census_income_set1.csv")
    print(f"Number of Rows and Columns {df.shape}")

    #convert dataframe into json so that we can dump these dataset into mongodb database
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # insert converted json record to mongodbgit 
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

