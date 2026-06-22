import pandas as pd

class InsightGenerator:

    def __init__(self,model,features_names):
        self.model = model
        self.features_names = features_names

    def get_feature_importance(self):

        # Tree based models
        if hasattr(
            self.model,
            "feature_importances_"
        ):

            importance = (
                self.model.feature_importances_
            )

        # Linear models
        elif hasattr(
            self.model,
            "coef_"
        ):

            importance = abs(
                self.model.coef_[0]
            )

        else:

            return None

        importance_df = pd.DataFrame({

            "Feature":
                self.features_names,

            "Importance":
                importance

        })

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
        )

        return importance_df
    
    def generate_business_summary(self):

        importance_df = self.get_feature_importance()

        top_features = (
            importance_df["Feature"]
            .head(5)
            .tolist()
        )

        summary = f"""
        Top factors influencing prediction:

        1. {top_features[0]}
        2. {top_features[1]}
        3. {top_features[2]}
        4. {top_features[3]}
        5. {top_features[4]}

        These features contribute most strongly
        to the model's decision-making process.
        """

        return summary
    
    def get_top_features(self, top_n=5):

        importance_df = self.get_feature_importance()

        return (
            importance_df["Feature"]
            .head(top_n)
            .tolist()
        )