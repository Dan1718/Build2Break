import csv

csv_file_path = "./AI_Human.csv"

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)
    print("CSV headers:", headers)
