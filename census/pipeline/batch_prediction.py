from census.exception import SensorException
from census.logger import logging
from census.predictor import ModelResolver
import pandas as pd
from census.utils import load_object
import os,sys
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
PREDICTION_DIR="prediction"

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)

        df.replace(to_replace="na",value=np.NAN,inplace=True)
        # replace ' ?' with np.NAN
        df.replace(to_replace=' ?',value=np.NAN,inplace = True)
        df.replace(to_replace='?',value=np.NAN,inplace = True)

        # remove space from string data 
        # like ' abc' --> 'abc'
        column_object = df.select_dtypes(include = 'object').columns
        for i in column_object:
            df[i] = df[i].str.strip()

        df["workclass"] = df["workclass"].replace(to_replace=np.NAN,value=df["workclass"].mode()[0])
        df["occupation"] = df["occupation"].replace(to_replace=np.NAN,value=df["occupation"].mode()[0])
        df["country"] = df["country"].replace(to_replace=np.NAN,value=df["country"].mode()[0])


        df = df.drop(['relationship'],axis = 1)
        for col in df.columns[:-1]:
            if df[col].dtypes == 'object':
                feature_test_label_encoder = LabelEncoder()
                df[col] = feature_test_label_encoder.fit_transform(df[col])

        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        
        cat_prediction = target_encoder.inverse_transform(prediction.astype('int'))

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise SensorException(e, sys)
