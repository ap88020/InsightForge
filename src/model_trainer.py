import joblib

from sklearn.model_selection import train_test_split

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    r2_score
)

class ModelTrainer:
    def __init__(self,df,target_column):
        
        self.df = df
        self.target_column = target_column

        self.X = None
        self.y = None

    def prepare_data(self):

        self.X = self.df.drop(
            columns=[self.target_column]
        )

        self.y = self.df[self.target_column]
    
    def detect_problem_type(self):
        
        unique_values = self.y.nunique()

        if self.y.dtype == "object":
            return "classification"
        elif unique_values <= 20:
            return "classification"
        else:
            return "regression"
    
    def train_classification_models(self):
        X_train,X_test,y_train,y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

        models = {
            "LogisticRegression":LogisticRegression(),
            "RandomForestClassifier":RandomForestClassifier(
                random_state=42
            )
        }

        results = {}

        for name , model in models.items():

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

            accuracy =  accuracy_score(
                y_test,
                prediction
            )
        
            results[name] = {
                "score": accuracy,
                "model": model
            }
        
        return results
    
    def train_regression_models(self):

        X_train,X_test,y_train,y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

        models = {
            "LinearRegression":LinearRegression(),
            "RandomForestRegressor":RandomForestRegressor(
                random_state=42
            )
        }
        
        results = {}
    
        for name , model in models.items():

            model.fit(X_train,y_train)

            predictions = model.predict(
                X_test
            )

            score = r2_score(
                y_test,
                predictions
            )

            results[name] = {
                "score":score,
                "model":model
            }
        
        return results
    
    def train(self):
        self.prepare_data()

        problem_type = self.detect_problem_type()

        print(
            f"Detect Problem Type : {problem_type}"
        )

        if problem_type == "classification":
            return self.train_classification_models()
        else:
            return self.train_regression_models()