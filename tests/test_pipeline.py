import sys
import os

project_root = os.path.abspath(
os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda import EDA
from src.visualizer import Visualizer
from src.model_trainer import ModelTrainer

loader = DataLoader("data/employe.csv")

df = loader.load_data()

cleaner = DataCleaner()

clean_df = cleaner.clean_data(
    df,
    target_column="Attrition"
)
eda = EDA(clean_df)

# visualizer = Visualizer(df)

# visualizer.generate_visualization()

trainer = ModelTrainer(
    clean_df,
    target_column="Attrition"
)

results = trainer.train()

print(results)

# print(clean_df["Attrition"].head())
# print(clean_df["Attrition"].unique())