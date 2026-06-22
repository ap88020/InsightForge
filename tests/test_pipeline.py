import sys
import os
from pprint import pprint

project_root = os.path.abspath(
os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda import EDA
from src.visualizer import Visualizer
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector

loader = DataLoader("data/employe.csv")

df = loader.load_data()

cleaner = DataCleaner()

clean_df = cleaner.clean_data(
    df,
    target_column="Attrition"
)
eda = EDA(clean_df)

trainer = ModelTrainer(
    clean_df,
    target_column="Attrition"
)

results = trainer.train()

# pprint(results)

selector = ModelSelector(results)

best_result = selector.select_best_model()

# pprint(best_result)

print("\n🏆 BEST MODEL")
print(best_result["best_model_name"])

print("\n📊 SCORE")
print(round(best_result["best_score"], 4))