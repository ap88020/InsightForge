import pandas as pd

class EDA:
    
    def __init__(self,df):
        self.df = df
    
    def dataset_summary(self):
        
        summary = {
            "rows" : self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplcate_values": int(self.df.duplicated().sum())
        }

        return summary
    
    def numerical_summary(self):
        
        numerical_df = self.df.select_dtypes(
            include=["int64","float64"]
        )

        return numerical_df.describe()
    
    def categorcal_summary(self):

        categorical_df = self.df.select_dtypes(
            include=["object","string"]
        )

        if categorical_df.empty:
            return "No categorical columns found."

        return categorical_df.describe()
    
    def correlation_analysis(self):
        
        numerical_df = self.df.select_dtypes(
            include=["int64","float64"]
        )

        print(numerical_df.dtypes)
        print(numerical_df.shape)

        return numerical_df.corr()
    
    def unique_values(self):
        
        unique_counts = {}

        for col in self.df.columns:
            unique_counts[col] = self.df[col].nunique()
        
        return unique_counts
    
    def data_types(self):
        return self.df.dtypes.astype(str)
    
    def generate_report(self):

        report = {
            "dataset_summary ":self.dataset_summary(),
            "data_types ":self.data_types(),
            "unique_values":self.unique_values(),
            "numerical_summary":self.numerical_summary(),
            "categorical_summary":self.categorcal_summary(),
            "corelation_matrix":self.correlation_analysis()
        }

        return report
