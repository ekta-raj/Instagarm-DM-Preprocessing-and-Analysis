# Instagram Data Preprocessing and Analysis Toolkit

This repository contains tools for preprocessing, de-identifying, and analyzing Instagram Direct Messages (DMs) for research and data-driven insights. The project includes two primary scripts that work together to streamline the processing and analysis of Instagram DM data.

---

### 1. **Instagram DM Preprocessor (`instagram_DM_preprocess.py`)**
   - **Purpose**: Preprocess and de-identify Instagram DM data from JSON and HTML files for secure and structured analysis.
   - **Key Features**:
     - **De-identification**: Utilizes the Presidio library to anonymize sensitive personal information (PII), such as email addresses, phone numbers, and names.
     - **Preprocessing**:
       - Converts emojis to text.
       - Anonymizes URLs and mentions.
       - Normalizes timestamps to UTC.
     - **Structured Output**: Consolidates messages into a unified structure, saved as Parquet files for efficient storage and analysis.
     - **Error Tracking**: Logs processing errors for traceability.
   - **Usage**: This script processes Instagram activity zip files, extracts and cleans data, and outputs structured, anonymized datasets.

---

### 2. **Instagram Data Analyzer (`data_analysis_instagram.py`)**
   - **Purpose**: Perform sentiment analysis and circadian rhythm analysis on preprocessed Instagram DM data.
   - **Key Features**:
     - **Sentiment Analysis**:
       - Uses VADER sentiment analysis to classify messages as Positive, Neutral, or Negative.
       - Generates a stacked bar chart visualizing sentiment distribution for each participant.
     - **Circadian Rhythm Analysis**:
       - Analyzes message activity by time of day across different time zones.
       - Produces a circadian rhythm chart showing participants' active hours.
     - **Metadata Management**: Stores analysis results and charts in a metadata file for reuse.
   - **Usage**: Reads Parquet files from the preprocessor and outputs visual insights as base64-encoded charts.

---

## Workflow

1. **Preprocessing**:
   - Use `instagram_DM_preprocess.py` to process Instagram DM data.
   - The script anonymizes sensitive data and structures the output for analysis.

   **Command**:
   ```bash
   python instagram_DM_preprocess.py --base_directory <input_directory> --output_directory <output_directory> --zip_file_prefix <prefix>
   ```

2. **Analysis**:
   - Use `data_analysis_instagram.py` to analyze the preprocessed data.
   - Generate sentiment and circadian rhythm charts.

   **Command**:
   ```bash
   python data_analysis_instagram.py <folder_path> <method> <metadata_file_path>
   ```

---

## Outputs

- **Preprocessor**:
  - Anonymized Parquet files with structured Instagram DM data.
  - Error logs detailing any processing issues.

- **Analyzer**:
  - Sentiment bar chart (stacked by Positive, Neutral, and Negative proportions).
  - Circadian rhythm chart showing normalized message activity over 24 hours.

---

## Dependencies

- **Core Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `scipy`
- **Sentiment Analysis**:
  - `vaderSentiment`
- **De-identification**:
  - `presidio-analyzer`, `presidio-anonymizer`
- **Data Storage**:
  - `pyarrow`, `parquet`
- **Other**:
  - `BeautifulSoup`, `emoji`, `pytz`, `argparse`

Install dependencies via pip:
```bash
pip install pandas numpy matplotlib scipy vaderSentiment presidio-analyzer presidio-anonymizer pyarrow beautifulsoup4 emoji pytz
```
