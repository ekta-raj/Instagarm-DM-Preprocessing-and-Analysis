import os
import re
import pandas as pd
import hashlib
import uuid
import zipfile
from datetime import datetime
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import argparse

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Regex patterns
URL_PATTERN = r'(https?://\S+)'
CONVERSATION_PATTERN = r'>>> Chat History with (.+?)::'
MESSAGE_PATTERN = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (.+?): (.+)'

# Hashing function for anonymization
def anonymize_sender(value):
    return value.encode('utf-8').hex() if value else 'unknown_sender'

# Generate random Conversation_ID in some cases where it does not mention who the Chat is with
def generate_random_conversation_id():
    return hashlib.sha256(str(uuid.uuid4()).encode('utf-8')).hexdigest()

# Message anonymization using Presidio
def anonymize_message(message):
    results = analyzer.analyze(
        text=message,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION"],
        language="en"
    )
    operators = {
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
        "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "[CITY]"}),
    }
    return anonymizer.anonymize(text=message, analyzer_results=results, operators=operators).text

# Standardize UTC timestamps
def standardize_utc(timestamp):
    try:
        datetime_obj = datetime.strptime(timestamp, "%H:%M:%S")
        return datetime_obj.strftime("%H:%M:%S")
    except Exception:
        return timestamp

# Removing URLs from the DM
def split_urls_and_text(lines):
    text_without_urls = []
    for line in lines:
        match = re.search(URL_PATTERN, line)
        if not match:
            text_without_urls.append(line.strip())
    return text_without_urls

# Process text lines into Parquet directly
def process_lines_to_parquet(lines, output_file):
    data = []
    conversation_id = None

    for line in lines:
        line = line.strip()

        if line.startswith(">>> Chat History with"):
            match = re.search(CONVERSATION_PATTERN, line)
            if match:
                conversation_id = anonymize_sender(match.group(1))
            else:
                conversation_id = generate_random_conversation_id()
            continue

        match = re.match(MESSAGE_PATTERN, line)
        if match:
            date, timestamp, sender, message = match.groups()
            sender_id = anonymize_sender(sender)
            utc_timestamp = standardize_utc(timestamp)
            anonymized_message = anonymize_message(message)

            full_utc_timestamp = f"{date} {utc_timestamp}"
            message_id = hashlib.sha256(f"{conversation_id}{full_utc_timestamp}{message}".encode('utf-8')).hexdigest()

            data.append({
                "Message_ID": message_id,
                "UTC_Timestamp": full_utc_timestamp,
                "Sender_ID": sender_id,
                "Message": anonymized_message,
                "Conversation_ID": conversation_id
            })

    df = pd.DataFrame(data, columns=["Message_ID", "UTC_Timestamp", "Sender_ID", "Message", "Conversation_ID"])
    df.to_parquet(output_file, index=False, engine="pyarrow")
    print(f"Processed data saved to: {output_file}")

# Main function
def process_zip_to_parquet(zip_file_path, output_file):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        text_file_name = next((name for name in file_names if "Direct Messages.txt" in name), None)
        
        if not text_file_name:
            raise FileNotFoundError("Direct Messages.txt not found in the ZIP file.")

        with zip_ref.open(text_file_name) as file:
            lines = file.read().decode('utf-8').splitlines()

    # Remove URLs and process remaining text
    text_without_urls = split_urls_and_text(lines)

    # Process the filtered text into a Parquet file
    process_lines_to_parquet(text_without_urls, output_file)

def main():
    parser = argparse.ArgumentParser(description="Process TikTok direct messages from ZIP to Parquet format")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input ZIP file containing TikTok data"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path for the output Parquet file"
    )
    
    args = parser.parse_args()
    process_zip_to_parquet(args.input, args.output)

if __name__ == "__main__":
    main()
