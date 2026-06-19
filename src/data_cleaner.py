import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    
    def __init__(self):
        self.label_encoder = {}
        self.scaler = StandardScaler()
    
    def handle_missing_values(self,df):

        num_cols = df.select_dtypes(
            include=["int64","float64"]
        ).columns

        cat_cols = df.select_dtypes(
            include=["object","string"]
        ).columns

        # Numerical column fill with median

        for col in num_cols:
            df[col] = df[col].fillna(
                df[col].median()
            )
        
        # Categorical column fill with mode

        for col in cat_cols:
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )
        
        return df
    
    def remove_duplicates(self,df):

        df = df.drop_duplicates()

        return df
    
    def encode_categorical(self,df):
        
        cat_cols = df.select_dtypes(
            include=["object","string"]
        ).columns

        for col in cat_cols:
            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(df[col])

            self.label_encoder[col] = encoder
        
        return df
    
    def scale_numerical(self,df):
        num_cols = df.select_dtypes(
            include=["int64","float64"]
        )

        for col in num_cols:
            df[col] = self.scaler.fit_transform(
                df[num_cols]
            )
        

        return df


    def clean_data(self,df):
        
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.encode_categorical(df)
        df = self.scale_numerical(df)

        return df
