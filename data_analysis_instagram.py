import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import argparse
import matplotlib.pyplot as plt
from typing import Dict, Callable, List, Optional
import pytz
import json
import base64
from io import BytesIO

class DataProcessor:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def perform_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Message' not in df.columns:
            print("'Message' column not found. Adding empty 'vader_compound_score' column.")
            df['vader_compound_score'] = pd.Series(dtype='float64')
        else:
            df['vader_compound_score'] = df['Message'].apply(self._analyze_sentiment)
        return df

    def _analyze_sentiment(self, text: str) -> float:
        if pd.isna(text):
            return 0.0
        return self.analyzer.polarity_scores(text)['compound']

class ChartCreator:
    @staticmethod
    def create_sentiment_bar_chart(data_dict: Dict[str, pd.DataFrame]) -> str:
        sentiment_data = []
        
        for participant, df in data_dict.items():
            df['sentiment'] = df['vader_compound_score'].apply(ChartCreator._categorize_sentiment)
            counts = df['sentiment'].value_counts(normalize=True)
            sentiment_data.append({
                'Participant': participant,
                'Negative': counts.get('Negative', 0),
                'Neutral': counts.get('Neutral', 0),
                'Positive': counts.get('Positive', 0)
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('Participant', inplace=True)
        
        color_map = {'Negative': 'red', 'Neutral': 'blue', 'Positive': 'green'}
        ax = sentiment_df.plot(kind='bar', stacked=True, figsize=(12, 6), color=[color_map[sentiment] for sentiment in sentiment_df.columns])
        
        plt.title('Sentiment Distribution by Participant')
        plt.xlabel('Participant')
        plt.ylabel('Proportion of Messages')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return ChartCreator._get_base64_image()

    @staticmethod
    def create_circadian_chart(data_dict: Dict[str, pd.DataFrame], timezone_dict: Dict[str, str]) -> str:
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(data_dict)))
        
        timezone_mapping = {
            'PT': 'US/Pacific',
            'MT': 'US/Mountain',
            'CT': 'US/Central',
            'ET': 'US/Eastern'
        }
        
        for (participant, df), color in zip(data_dict.items(), colors):
            df['datetime'] = pd.to_datetime(df['UTC_Timestamp'])
            
            timezone = timezone_dict.get(participant, 'UTC')
            full_timezone = timezone_mapping.get(timezone, timezone)
            try:
                tz = pytz.timezone(full_timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"Unknown timezone '{timezone}' for participant {participant}. Using UTC.")
                tz = pytz.UTC
            
            df['adjusted_time'] = df['datetime'].dt.tz_localize(pytz.UTC).dt.tz_convert(tz)
            df['time_of_day'] = df['adjusted_time'].dt.hour + df['adjusted_time'].dt.minute / 60
            
            participant_data = df[df['expected_sender']]

            kde = gaussian_kde(participant_data['time_of_day'])
            x_range = np.linspace(0, 24, 240)
            y_kde = kde(x_range)
            y_kde_normalized = y_kde / y_kde.max()

            plt.plot(x_range, y_kde_normalized, color=color, label=f"{participant} ({timezone})")
            plt.fill_between(x_range, 0, y_kde_normalized, color=color, alpha=0.3)

        plt.xlabel('Time of Day (Hours)')
        plt.ylabel('Normalized Activity')
        plt.title('Overlapping Circadian Rhythm Chart of Messages')
        plt.xlim(0, 24)
        plt.xticks(range(0, 25, 2))
        plt.ylim(0, 1.1)  # Set y-axis limit to accommodate all curves
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return ChartCreator._get_base64_image()

    _categorize_sentiment = staticmethod(lambda score: 'Negative' if score <= -0.05 else 'Positive' if score >= 0.05 else 'Neutral')

    @staticmethod
    def _get_base64_image():
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class Pipeline:
    def __init__(self, folder_path: str, processor: DataProcessor, method: Callable, metadata_file_path: str):
        self.folder_path = folder_path
        self.processor = processor
        self.method = method
        self.failure_log = os.path.join(folder_path, "failure_log.txt")
        self.processed_data: Dict[str, pd.DataFrame] = {}
        self.metadata_file_path = metadata_file_path
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        try:
            with open(self.metadata_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"participant_level_data": {}, "project_level_data": {}}

    def _save_metadata(self):
        with open(self.metadata_file_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def process_folder(self, return_data: bool = False) -> Optional[List[pd.DataFrame]]:
        for file in os.listdir(self.folder_path):
            #extract the name 
            file_name = file.split(".")[0]
            file_path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_parquet(file_path)
                df = self.method(df)
                df.to_parquet(file_path, index=False)
                if df is not None:
                    self.processed_data[file_name] = df
            except Exception as e:
                with open(self.failure_log, 'a') as f:
                    f.write(f"Failed to process {file_path}: {str(e)}\n")

        if return_data:
            return self.processed_data
        
        return None


#TODO: should ideally be in a separate file     
def main():
    parser = argparse.ArgumentParser(description="Process Parquet or CSV files in a folder.")
    parser.add_argument("folder_path", help="Path to the folder containing Parquet or CSV files")
    parser.add_argument("method", help="Method to be applied to the data")
    parser.add_argument("metadata_file_path", help="Path to the metadata JSON file")
    args = parser.parse_args()

    methods = {"perform_sentiment_analysis": DataProcessor().perform_sentiment_analysis}
    processor = DataProcessor()
    pipeline = Pipeline(args.folder_path, processor, methods[args.method], args.metadata_file_path)
    pipeline.process_folder()
    
    sample_chart_creator = ChartCreator()
    sentiment_chart = sample_chart_creator.create_sentiment_bar_chart(pipeline.processed_data)
    circadian_chart = sample_chart_creator.create_circadian_chart(pipeline.processed_data, pipeline.metadata["participant_level_data"]["timezone"])
    
    # Save the charts in the metadata
    pipeline.metadata["project_level_data"]["sentiment_chart"] = sentiment_chart
    pipeline.metadata["project_level_data"]["circadian_chart"] = circadian_chart

    pipeline._save_metadata()

if __name__ == "__main__":
    main()
