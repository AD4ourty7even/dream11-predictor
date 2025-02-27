import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from clean import clean_data

class BlankPageException(Exception):
    pass



def scrape_cricinfo_batting_stats(url):
    # Send HTTP request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',        
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the table body (<tbody>) tag
    tbody = soup.find('tbody')
    if not tbody:
        print("No <tbody> found on the page.")
        return None

    rows = []
    data_rows = tbody.find_all('tr')

    for tr in data_rows:
        row = []
        for cell in tr.find_all('td'):
            row.append(cell.text.strip())

        if row:  # Skip empty rows
            rows.append(row)

    if not rows:
        print("No data rows found in the <tbody> section.")
        return None

    # Extract headers (using the first row in <thead> or a separate header row if available)
    header_row = soup.find('thead')
    headers = []

    if header_row:
        for cell in header_row.find_all('th'):
            headers.append(cell.text.strip())

    if not headers:
        raise BlankPageException("No headers found in the table.")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


def scrape_multiple_pages(start_date, end_date, stats_type):
    start_date = start_date.replace(' ', '+')
    end_date = end_date.replace(' ', '+')
    base_url = build_cricinfo_url(start_date, end_date, stats_type)
    
    total_pages = 100
    
    all_data = []
    for page in range(1, total_pages + 1):
        try:
            print(f"Scraping page {page}...")
            url = base_url.replace("page=1", f"page={page}")
            df = scrape_cricinfo_batting_stats(url)
            time.sleep(0.5)
            if df is not None:
                print(f"Successfully scraped data from page {page}.")
                all_data.append(df)
            else:
                print(f"Failed to scrape data from page {page}.")
        except BlankPageException as e:
            print(e)
            break
    # Combine all pages into a single DataFrame
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        export_to_csv(combined_df, f'cricinfo_{stats_type}_stats.csv')
        
        return combined_df
    else:
        return None

def export_to_csv(df: pd.DataFrame, filename='cricinfo_bowling_stats.csv'):
    df.to_csv(filename, index=False, mode='w+')
    
    print(f"Data exported to {filename}")

def build_cricinfo_url(start_date, end_date, stats_type='batting'):
    valid_types = {'batting', 'fielding', 'bowling'}
    if stats_type.lower() not in valid_types:
        raise ValueError("stats_type must be either 'batting' or 'fielding'")

    base_url = (
        "https://stats.espncricinfo.com/ci/engine/stats/index.html?"
        "class=2;filter=advanced;orderby=batted_score;page=1;size=200;"
        f"spanmax2={end_date};spanmin2={start_date};spanval2=span;"
        "team=1;team=2;team=25;team=3;team=40;team=5;team=6;team=7;"
        f"template=results;type={stats_type.lower()};view=innings;"
    )
    return base_url


def merge_dfs(df1, df2, df3):
    def clean_columns(df, suffix):
        # Reset index and clean column names
        df = df.reset_index(drop=True)
        df.columns = [col.strip().lower().replace(' ', '_') + suffix for col in df.columns]
        return df
    
    # Clean and prepare DataFrames
    df1 = clean_columns(df1, '_1')
    df2 = clean_columns(df2, '_2')
    df3 = clean_columns(df3, '_3')
    
    # First merge
    try:
        final_df = pd.merge(df1, df2,
                           left_on=['player_1', 'opposition_1', 'ground_1', 'start_date_1'],
                           right_on=['player_2', 'opposition_2', 'ground_2', 'start_date_2'],
                           how='outer',
                           validate=None)  # Disable validation temporarily
        
        # Second merge
        final_df = pd.merge(final_df, df3,
                           left_on=['player_1', 'opposition_1', 'ground_1', 'start_date_1'],
                           right_on=['player_3', 'opposition_3', 'ground_3', 'start_date_3'],
                           how='outer',
                           validate=None)  # Disable validation temporarily
        
        # Reset index after merges
        final_df = final_df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error during merge: {str(e)}")
        # Print the shapes and column names for debugging
        print(f"df1 shape: {df1.shape}, columns: {df1.columns}")
        print(f"df2 shape: {df2.shape}, columns: {df2.columns}")
        print(f"df3 shape: {df3.shape}, columns: {df3.columns}")
        raise
    
    # Rest of the function remains the same...
    columns_to_drop = ['unnamed:_8_1', 'unnamed:_7_2', 'ground_1','opposition_1', 'start_date_1',
                    'player_2', 'opposition_2', 'start_date_2', 'ground_2', 'unnamed:_11_2',
                    'player_3', 'unnamed:_7_3', 'unnamed:_11_3', 'inns_3']
    final_df = final_df.drop(columns=columns_to_drop, errors='ignore')
    
    final_df.columns = final_df.columns.str.replace(r'(_\d+)', '', regex=True)
    final_df = final_df.rename(columns={'inns': 'inns_batting', 'inns_2': 'inns_bowling'})
    final_df['start_date'] = pd.to_datetime(final_df['start_date'], format='%d %b %Y')
    final_df = final_df.sort_values(by='start_date', ascending=False)
    final_df[['player', 'country']] = final_df['player'].str.extract(r'([^\(]+)\s\(([^)]+)\)')
    
    return final_df

def main():
    start_date='17 Feb 2015'
    end_date='25 Feb 2025'

    
    for stats_type in ['batting', 'bowling', 'fielding']:
        scrape_multiple_pages(start_date, end_date, stats_type)

    
    df1 = pd.read_csv('cricinfo_batting_stats.csv')
    df2 = pd.read_csv('cricinfo_bowling_stats.csv')
    df3 = pd.read_csv('cricinfo_fielding_stats.csv')

    # Function to clean column names (convert to lowercase, replace spaces with underscores)
    
    final_df = merge_dfs(df1, df2, df3)
    # Clean column names and add suffixes
    
    final_df = clean_data(final_df)

    final_df.to_csv('processed_final.csv')




# main()

# Load the three CSV files into DataFrames
