import pandas as pd

# Define fantasy points calculation function
def calculate_fantasy_points(row):
    points = 4
    
    # Batting points
    if row['runs'] not in ['sub', 'TDNB', 'DNB', 'absent']:
        if not pd.isna(row['runs_clean']):
            # Check for duck (0 runs) for non-bowlers
            # Only count as duck if player is out (no * in runs) and is not a bowler
            if row['runs_clean'] == 0 and row['Player Type'] != 'BOWL' and '*' not in str(row['runs']):
                points -= 3
            if not pd.isna(row['runs_clean']):
                runs = float(row['runs_clean'])
                points += runs  # 1 point per run
            
            # Bonus points for 50s and 100s
            if runs >= 25:
                points += 4
            if runs >= 50:
                points += 8
            if runs >= 75:
                points += 12
            if runs >= 100:
                points += 16
            if runs >= 125:
                points += 20
            if runs >= 150:
                points += 24
                
            # Points for strike rate
            if not pd.isna(row['sr']) and not pd.isna(row['bf']) and float(row['bf']) >= 20:
                sr = float(row['sr'])
                if sr >= 140:
                    points += 6
                elif sr > 120:
                    points += 4
                elif sr > 100:
                    points += 2
                elif sr>=40 and sr<=50:
                    points -= 2
                elif sr>=30 and sr<40:
                    points -= 4
                elif sr<30:
                    points -= 6
                    
            # Points for 4s and 6s        
            if not pd.isna(row['4s']):
                points += float(row['4s']) * 4
            if not pd.isna(row['6s']):    
                points += float(row['6s']) * 6
    
    # Bowling points
    if row['overs'] not in ['sub', 'TDNB', 'DNB', 'absent']:
        if not pd.isna(row['wkts']):
            points += float(row['wkts']) * 25
            
            # Bonus for 3+ wickets
            if float(row['wkts']) >=4:
                points += 4
            elif float(row['wkts']) >= 5:
                points += 8
            elif float(row['wkts']) >= 6:
                points += 12
                
        # Economy rate points        
        if not pd.isna(row['econ']) and not pd.isna(row['overs']) and float(row['overs']) >= 5:
            econ = float(row['econ'])
            if econ < 2.5:
                points += 6
            elif econ < 3.5:
                points += 4
            elif econ < 4.5:
                points += 2
            elif econ >= 7 and econ <= 8:
                points -= 2
            elif econ >8 and econ <=9:
                points -= 4
            elif econ > 9:
                points -= 6
        
        if not pd.isna(row['mdns']):
            points += float(row['mdns'])*6
                
    # Fielding points            
    if not pd.isna(row['ct']):
        ct = float(row['ct'])
        points += ct * 8
        if ct >= 3:
            points += 4
    if not pd.isna(row['st']):    
        points += float(row['st']) * 12
        
    return points


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the unnamed column (if it exists)
    df = df.drop('unnamed:', axis=1, errors="ignore")
    
    # First get list of column names
    cols = df.columns.tolist()
    
    # Find the indices of duplicate columns
    inns_indices = [i for i, col in enumerate(cols) if col == 'inns_batting']
    runs_indices = [i for i, col in enumerate(cols) if col == 'runs']
    
    # Rename second occurrence of columns
    if len(inns_indices) > 1:
        cols[inns_indices[1]] = 'inns_bowling'
    
    cols[runs_indices[1]] = 'runs_given'
    df.columns = cols

    # Create a new column 'runs_clean' by removing '*' from runs values
    df['runs_clean'] = df['runs'].astype(str).str.replace('*', '')

    # Convert DNB and other non-numeric values to NaN
    df['runs_clean'] = pd.to_numeric(df['runs_clean'], errors='coerce')

    # Read the mapping CSV file that contains player name mappings and additional data
    player_mapping_df = pd.read_csv('CT_SquadData_AllTeams.csv')

    # Assuming the player names in the current df are in 'player' column
    # And player names in mapping file are in 'original_name' column
    # Merge the dataframes based on player names
    df = pd.merge(
        df,
        player_mapping_df,
        left_on='player',  # Column name in current df
        right_on='Original name',  # Column name in mapping df
        how='left'  # Keep all rows from original df even if no match found
    )

    # Drop the duplicate name column if needed
    if 'Original name' in df.columns:
        df = df.drop('Original name', axis=1)

    # Convert '-' to 0 for numeric columns
    numeric_columns = ['runs', 'mins', 'bf', '4s', '6s', 'sr', 'mdns', 'runs_given', 'wkts', 'econ', 'dis', 'ct', 'st', 'ct_wk', 'ct_fi', 'runs_clean']

    for col in numeric_columns:
        df[col] = df[col].replace('-', '0')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['fantasy_points'] = df.apply(calculate_fantasy_points, axis=1)

    ground_details = pd.read_csv('ground_details.csv')

    df = df.merge(
        ground_details[['ground_name', 'preferred_style']], 
        left_on='ground',
        right_on='ground_name',
        how='left'
    )

    df = df.drop('ground_name', axis=1)

    ct_squad_players = player_mapping_df['Original name'].unique()

    # Filter cricket data to only include players in CT squad
    df = df[df['player'].isin(ct_squad_players)]

    # Merge bowling style and Player name column from squad_final.csv to processed_cricket_data1.csv
    # df = pd.merge(df, player_mapping_df[['Original name', 'Bowling Style']], left_on='player', right_on='Original name', how='left')

    # Drop the redundant Player name column after merge
    # df = df.drop('Original name', axis=1)

    # Define opponent tier mapping
    tier_mapping = {
        "v Australia": "1", "v India": "1", "v England": "1",
        "v South Africa": "2", "v Pakistan": "2", "v New Zealand": "2", "v Sri Lanka": "2",
        "v West Indies": "3", "v Bangladesh": "3", "v Afghanistan": "3", "v Zimbabwe": "3", "v Ireland": "3",
        "v Netherlands": "4", "v Scotland": "4", "v Nepal": "4", "v U.A.E.": "4", "v Hong Kong": "4"
    }

    # Add new column based on opponent tier
    df['opponent_tier'] = df['opposition'].map(tier_mapping).fillna("Unknown")

    return df




