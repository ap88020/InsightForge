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

        fig, ax = plt.subplots(figsize=(10,5))

        missing.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Missing Values")

        plt.tight_layout()

        return fig

    def plot_histograms(self):

        numerical_cols = self.df.select_dtypes(
            include=["int64","float64"]
        ).columns

        figs = []

        for col in numerical_cols:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.histplot(
                self.df[col],
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"{col} Distribution"
            )

            figs.append(fig)

        return figs
        

    def plot_boxplots(self):

        numerical_cols = self.df.select_dtypes(
            include=["int64","float64"]
        ).columns

        figs = []

        for col in numerical_cols:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.boxplot(
                x=self.df[col],
                ax=ax
            )

            ax.set_title(
                f"{col} Boxplot"
            )

            figs.append(fig)

        return figs
        

    def plot_correlation_heatmap(self):

        numerical_df = self.df.select_dtypes(
            include=["int64","float64"]
        )

        numerical_df = numerical_df.loc[
            :,
            numerical_df.nunique() > 1
        ]

        corr = numerical_df.corr()

        fig, ax = plt.subplots(figsize=(12,8))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        return fig

    def generate_visualization(self):
        
        self.plot_missing_values()
        self.plot_histograms()
        self.plot_boxplots()
        self.plt_correlation_heatmap()
    