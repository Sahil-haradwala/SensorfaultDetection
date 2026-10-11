from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://sahilharadwala123_db_user:C1DTgT5sznORnxzA@sensorfault.oz0wme4.mongodb.net/?appName=Sensorfault"
client = MongoClient(uri)
MONGO_DATABASE_NAME = "Sensorfault"
MONGO_COLLECTION_NAME = "Waferfault"

df = pd.read_csv("notebook/wafer_23012020_041211.csv")

df = df.drop("Unnamed", axis=1)

json_record = list(json.loads(df.T.to_json()).values())

client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME].insert_many(json_record)
print("Data uploaded successfully!")
