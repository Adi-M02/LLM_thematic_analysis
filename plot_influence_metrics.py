import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler


def plot_range_histogram(title, csv_file_path, min_val, max_val, bin_size):
    file_path = csv_file_path
    df = pd.read_csv(file_path)
        # Define bin edges
    # bin_edges = range(0, max(df['upvotes']) + 1000, 1000)
    bin_edges = range(min_val, max_val, bin_size)

    # Create bins and count the number of users in each bin
    df['bin'] = pd.cut(df[title], bins=bin_edges, right=False)
    bin_counts = df['bin'].value_counts(sort=False)

    # Create a DataFrame with bin ranges and user counts
    bin_counts_df = bin_counts.reset_index()
    bin_counts_df.columns = [f'{title} Range', 'Number of Users']

    # Extract only the upper bound of each range for the x-axis labels
    bin_counts_df[f'{title} Upper Bound'] = bin_counts_df[f'{title} Range'].apply(lambda x: x.right)

    # Plot the histogram
    plt.bar(bin_counts_df[f'{title} Upper Bound'], bin_counts_df['Number of Users'], width=90, edgecolor='black', align='center')
    plt.xticks(bin_counts_df[f'{title} Upper Bound'], rotation=45, ha='right')
    plt.title(f'Distribution of Users by Number of {title}')
    plt.xlabel(f'{title} Range Upper Bound')
    plt.ylabel('Number of Users')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_range_histogram("num_comments", "influence/comment_influence/OR_user_num_commented.csv", 0, 2000, 100)