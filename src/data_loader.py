import pandas as pd
from pprint import pprint

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("FileNotFoundError")
            return None
    
    def get_metadata(self,df):

        num_cols = df.select_dtypes(
            include=["int64","float64"]
        ).columns.tolist()
    
        cat_cols = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        metadata = {
            "rows":df.shape[0],
            "columns":df.shape[1],
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
        }


        return metadata
    
    def get_missing_values(self,df):
        miss_value = df.isnull().sum()
        miss_value = miss_value[miss_value > 0]

        return miss_value
    
    def get_duplicate_count(self,df):
        dup_value = df.duplicated().sum()
        return dup_value
    
    


# loader = DataLoader("employe.csv")

# df = loader.load_data() 


# if df is not None:
#     metadata = loader.get_metadata(df)
#     missdata = loader.get_missing_values(df)
#     dupdata = loader.get_duplicate_count(df)
#     pprint(metadata)
#     print("Missing values :" , missdata)
#     print("duplicated data",dupdata)
