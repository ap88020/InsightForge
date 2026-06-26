import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()


class LLMReportGenerator:
    
    def __init__(self,api_key):
        
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=api_key,
            temperature=0.3
        )

    def generate_report(self,dataset_summary,best_model,best_score,top_features):
        prompt = ChatPromptTemplate.from_template("""
            You are an expert Senior Data Scientist and Business Consultant.

            Analyze the following machine learning results and produce a professional business report.

            Dataset Summary:
            {dataset_summary}

            Best Machine Learning Model:
            {best_model}

            Model Performance:
            {best_score}

            Top Important Features:
            {top_features}

            Requirements:

            1. Executive Summary
            2. Dataset Overview
            3. Model Performance Analysis
            4. Feature Importance Analysis
            5. Business Insights
            6. Business Recommendations
            7. Risks & Limitations
            8. Final Conclusion

            Rules:
            - Explain the model performance in simple business language.
            - Explain why the important features matter.
            - Give practical recommendations.
            - Use Markdown headings.
            - Use bullet points where appropriate.
            - Keep the report between 500 and 800 words.
            - Do not simply repeat the input values.
        """)

        chain = prompt | self.llm

        response = chain.invoke({
            "dataset_summary" : dataset_summary,
            "best_model" : best_model,
            "best_score" : best_score,
            "top_features" : top_features
        })

        return response.content        