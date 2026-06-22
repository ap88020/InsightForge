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
        prompt = ChatPromptTemplate.from_template(
             """
                    You are a Senior Data Analyst.

                    Dataset Information:
                    {dataset_summary}

                    Best Model:
                    {best_model}

                    Model Score:
                    {best_score}

                    Top Important Features:
                    {top_features}

                    Generate:

                    1. Executive Summary
                    2. Key Insights
                    3. Business Recommendations
                    4. Potential Risks
                    5. Final Conclusion

                    Write professionally.
                    """
        )

        chain = prompt | self.llm

        response = chain.invoke({
            "dataset_summary" : dataset_summary,
            "best_model" : best_model,
            "best_score" : best_score,
            "top_features" : top_features
        })

        return response.content        