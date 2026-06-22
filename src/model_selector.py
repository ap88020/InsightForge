import joblib
import os

class ModelSelector:
    
    def __init__(self,results):
        self.results = results
    
    def select_best_model(self):
        best_model_name = None
        best_model = None
        best_score = float("-inf")

        for mode_name, result in self.results.items():
            if result["score"] > best_score:

                best_score = result["score"]

                best_model_name = mode_name

                best_model = result["model"]
            
            return {
                "best_model_name":best_model_name,
                "best_score":best_score,
                "best_model":best_model
            }
    
    def save_model(self,model,model_name):

        os.makedirs(
            "models",
            exist_ok=True
        )

        model_path = f"models/{model_name}.pkl"

        joblib.dump(
            model,
            model_path
        )

        return model_path