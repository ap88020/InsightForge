import sys
import os
from pprint import pprint
from dotenv import load_dotenv

project_root = os.path.abspath(
os.path.join(os.path.dirname(__file__), "..")
)

load_dotenv()


api_key = os.getenv("MISTRAL_API_KEY")

# print(api_key)

sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.eda import EDA
from src.visualizer import Visualizer
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.insight_generator import InsightGenerator
from src.llm_report_generator import LLMReportGenerator

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



best_model = (
    best_result["best_model"]
)

feature_name = (
    clean_df.drop(columns=["Attrition"]).columns
)

generator = InsightGenerator(best_model,feature_name)

# insights = (
#     generator.generate_business_summary()
# )


top_features = generator.get_top_features()

llm = LLMReportGenerator(
    api_key=api_key
)

report = llm.generate_report(
    dataset_summary=eda.dataset_summary(),
    best_model=best_result["best_model_name"],
    best_score=best_result["best_score"],
    top_features=top_features
)

# pprint(report)

os.makedirs(
    "reports/results",
    exist_ok=True
)

dataset_name = os.path.splitext(
    os.path.basename(loader.file_path)
)[0]

report_file = (
    f"reports/{dataset_name}_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

print(
    f"Report saved at: {report_file}"
)