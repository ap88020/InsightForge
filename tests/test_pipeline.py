import sys
import os

project_root = os.path.abspath(
os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda import EDA

loader = DataLoader("data/employe.csv")

df = loader.load_data()

cleaner = DataCleaner()

clean_df = cleaner.clean_data(df)

eda = EDA(clean_df)

print("REPORT")
print(eda.generate_report())
