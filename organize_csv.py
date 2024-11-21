import csv
import os

def save_ranges_to_csv(ranges, output_file, interval_size):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Range', 'Count'])
        for i, count in enumerate(ranges):
            writer.writerow([f'{i * interval_size}-{(i + 1) * interval_size - 1}', count])

def upvotes_to_ranges(csv_file, output_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        ranges = [0] * 
        interval_size = (10000000 + 9999) // 10000
        for row in reader:
            upvote_count = int(row[1])
            index = min(upvote_count // interval_size, 999)
            ranges[index] += 1
    save_ranges_to_csv(ranges, output_file, interval_size)


if __name__ == "__main__":
    upvotes_to_ranges('influence/user_upvotes.csv', 'influence/organized/upvote_ranges_1000.csv')
