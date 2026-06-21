import os
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:

    def __init__(self, df):
        self.df = df

        os.makedirs(
            "reports/plots",
            exist_ok=True
        )
    
    def plot_missing_values(self):
        missing = self.df.isnull().sum()

        plt.figure(figsize=(10,5))

        missing.plot(kind="bar")

        plt.title("Missing_values")

        plt.tight_layout()

        plt.savefig(
            "reports/graphs/missing_values.png"
        )

        plt.close()
    
    def plot_histograms(self):

        numerical_cols = self.df.select_dtypes(
            include=["int64","float64"]
        ).columns

        for col in numerical_cols:

            plt.figure(figsize=(6,4))

            sns.histplot(
                self.df[col],
                kde="True"
            )

            plt.title(
                f"{col} Distrubution"
            )

            plt.tight_layout()

            plt.savefig(
                f"reports/graphs/_hist.png"
            )

            plt.close()
        
    def plot_boxplots(self):

        numerical_col = self.df.select_dtypes(
            include=["int64","float64"]
        )

        for col in numerical_col:

            plt.figure(figsize=(6,4))

            sns.boxplot(
                x = self.df[col]
            )

            plt.title(f"{col} Boxplot")

            plt.tight_layout()

            plt.savefig(
                f"reports/graphs/{col}_boxplot.png"
            )

            plt.close()
        
    def plt_correlation_heatmap(self):

        numerical_df = self.df.select_dtypes(
            include=["int64","float64"]
        )

        numerical_df = numerical_df.loc[
            :,
            numerical_df.nunique() > 1
        ]

        corr_matrix = numerical_df.corr()

        plt.figure(
            figsize=(15,20)
        )

        sns.heatmap(
            corr_matrix,
            cmap="coolwarm"
        )

        plt.title(
            "Correlation Heatmap"
        )
        
        plt.tight_layout()

        plt.savefig(
            f"reports/graphs/correlation_heatmap.png"
        )    

        plt.close()

    def generate_visualization(self):
        
        self.plot_missing_values()
        self.plot_histograms()
        self.plot_boxplots()
        self.plt_correlation_heatmap()
    