import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from datetime import datetime, timedelta
from scrape import scrape_multiple_pages, clean_data, merge_dfs
warnings.filterwarnings('ignore')

def get_opponent_tier(team):
    """Returns the tier of a team based on the specified classification"""
    tier_mapping = {
        # Tier 1
        'AUS': 1, 'IND': 1, 'ENG': 1,
        
        # Tier 2
        'SA': 2, 'PAK': 2, 'NZ': 2, 'SL': 2,
        
        # Tier 3
        'WI': 3, 'BAN': 3, 'AFG': 3, 'ZIM': 3, 'IRE': 3,
        
        # Tier 4
        'NED': 4, 'SCO': 4, 'NEP': 4, 'UAE': 4, 'HK': 4
    }
    return tier_mapping.get(team, 4)

def clean_df(df):
    """Clean and preprocess the data"""
    df = df.copy()
    
    # Convert innings batting and related columns to numeric
    df = df[~df['inns_batting'].isin(['TDNB', 'sub'])]
    # df = df[~df['inns_batting.1'].isin(['TDNB', 'sub'])]
    df['inns_batting'] = pd.to_numeric(df['inns_batting'].replace(['DNB', '-'], '0'), errors='coerce')
    # df['inns_batting.1'] = pd.to_numeric(df['inns_batting.1'].replace(['DNB', '-'], '0'), errors='coerce')
    
    # Convert overs to numeric
    df = df[~df['overs'].isin(['TDNB', 'sub'])]
    df['overs'] = pd.to_numeric(df['overs'].replace(['DNB', '-'], '0'), errors='coerce')
    
    # Convert numeric columns
    numeric_columns = [
        'runs', 'mins', 'bf', '4s', '6s', 'sr', 
        'mdns', 'runs_given', 'wkts', 'econ',
        'ct', 'st', 'ct_wk', 'ct_fi', 'Credits',
        'fantasy_points'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Update opponent tier
    df['opposition'] = df['opposition'].str.replace('v ', '')
    df['opponent_tier'] = df['opposition'].map(get_opponent_tier)
    
    # Fill NaN values
    df['runs'] = df['runs'].fillna(0)
    df['wkts'] = df['wkts'].fillna(0)
    df['fantasy_points'] = df['fantasy_points'].fillna(0)
    
    # Handle categorical columns
    df['Bowling Style'] = df['Bowling Style'].fillna('NONE')
    df['preferred_style'] = df['preferred_style'].fillna('BAT')
    df['Player Type'] = df['Player Type'].fillna('BAT')
    
    return df

def calculate_weighted_features(df):
    """Calculate weighted features based on player type"""
    weights = {
        'BAT': {
            'runs': 1.5,
            '4s': 1.2,
            '6s': 1.3,
            'sr': 1.1,
            'ct_fi': 0.8
        },
        'BOWL': {
            'overs': 1.2,
            'mdns': 1.1,
            'runs_given': -0.8,
            'wkts': 1.5,
            'econ': -1.0,
            'ct_fi': 0.8
        },
        'ALL': {
            'runs': 1.2,
            '4s': 1.0,
            '6s': 1.1,
            'sr': 0.9,
            'overs': 1.0,
            'mdns': 0.9,
            'runs_given': -0.7,
            'wkts': 1.3,
            'econ': -0.8,
            'ct_fi': 0.8
        },
        'WK': {
            'runs': 1.2,
            '4s': 1.0,
            '6s': 1.1,
            'sr': 0.9,
            'st': 1.3,
            'ct_wk': 1.4
        }
    }
    
    for player_type, weight_dict in weights.items():
        mask = df['Player Type'] == player_type
        for feature, weight in weight_dict.items():
            if feature in df.columns:
                weighted_col = f'{feature}_weighted'
                df.loc[mask, weighted_col] = df.loc[mask, feature] * weight
    
    return df

def create_player_features(df):
    """Create features for prediction"""
    df = clean_df(df)
    
    # Convert start_date to datetime
    # print(df['start_date'])
    df['start_date'] = df['start_date'].str.split(' ').str[0]  # Remove timestamp
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Calculate time-based features
    df['days_since_match'] = (df.groupby('player')['start_date']
                             .transform(lambda x: (x.max() - x).dt.days))
    df['time_weight'] = np.exp(-df['days_since_match'] / 365)
    
    # Apply player type-specific weightages
    df = calculate_weighted_features(df)
    
    # Calculate weighted rolling averages
    df = df.sort_values(['player', 'start_date'])
    windows = [3, 5]
    
    for window in windows:
        # Batting features
        for feat in ['runs_weighted', '4s_weighted', '6s_weighted']:
            if feat in df.columns:
                df[f'{feat}_avg_{window}'] = (df.groupby('player')[feat]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean()))
        
        # Bowling features
        for feat in ['wkts_weighted', 'econ_weighted', 'mdns_weighted']:
            if feat in df.columns:
                df[f'{feat}_avg_{window}'] = (df.groupby('player')[feat]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean()))
        
        # Fielding features
        for feat in ['ct_fi_weighted', 'ct_wk_weighted', 'st_weighted']:
            if feat in df.columns:
                df[f'{feat}_avg_{window}'] = (df.groupby('player')[feat]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean()))
    
    # Opposition based features
    df['performance_vs_opposition'] = df.groupby(['player', 'opposition'])['fantasy_points'].transform('mean')
    df['performance_vs_tier'] = df.groupby(['player', 'opponent_tier'])['fantasy_points'].transform('mean')
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    return df

def select_features(df):
    """Select relevant features for the model"""
    base_features = [
        'days_since_match',
        'time_weight',
        'opponent_tier',
        'performance_vs_opposition',
        'performance_vs_tier'
    ]
    
    batting_features = [col for col in df.columns if 'runs_weighted' in col or '4s_weighted' in col or '6s_weighted' in col]
    bowling_features = [col for col in df.columns if 'wkts_weighted' in col or 'econ_weighted' in col or 'mdns_weighted' in col]
    fielding_features = [col for col in df.columns if 'ct_fi_weighted' in col or 'ct_wk_weighted' in col or 'st_weighted' in col]
    
    return base_features + batting_features + bowling_features + fielding_features

def predict_next_match(model, player_data, opponent_team, ground=None, pitch_type=None):
    """Make predictions for a player"""
    try:
        features = select_features(player_data)
        X_pred = player_data[features].iloc[-1:].copy()
        
        # Update prediction features
        last_match_date = player_data['start_date'].max()
        current_date = pd.Timestamp.now()
        days_since_last = (current_date - last_match_date).days
        
        X_pred['days_since_match'] = days_since_last
        X_pred['time_weight'] = np.exp(-days_since_last / 365)
        X_pred['opponent_tier'] = get_opponent_tier(opponent_team)
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        # Calculate confidence interval
        recent_std = np.std(player_data['fantasy_points'].tail(10))
        opponent_history = len(player_data[player_data['opposition'].str.replace('v ', '').map({'India': 'IND', 'Australia': 'AUS', 'England': 'ENG', 'Pakistan': 'PAK', 'South Africa': 'SA', 'New Zealand': 'NZ', 'Bangladesh': 'BAN', 'Afghanistan': 'AFG'}) == opponent_team])
        
        confidence_factor = 1 + (1 / (opponent_history + 1)) + (days_since_last / 365)
        ci_width = 1.96 * recent_std * confidence_factor
        
        confidence_interval = (
            max(0, prediction - ci_width),
            prediction + ci_width
        )
        
        return prediction, confidence_interval, {
            'opponent_tier': X_pred['opponent_tier'].iloc[0],
            'opponent_history': opponent_history,
            'days_since_last': days_since_last
        }
        
    except Exception as e:
        print(f"\nError predicting for {player_data['player'].iloc[0]}: {str(e)}")
        return None, None, None

def get_player_inputs():
    df_input = get_squad_data()
    teams = sorted(df_input['Team'].unique())
    team1, team2 = teams
    # Assuming 'df_input' is your DataFrame
    team1_players = df_input[(df_input['Team'] == team1) & (df_input['IsPlaying'] == 'PLAYING')]['Player Name']
    team2_players = df_input[(df_input['Team'] == team2) & (df_input['IsPlaying'] == 'PLAYING')]['Player Name']

    ground_details = pd.read_csv('ground_details.csv') 
    # Create a dictionary for stadium mapping
    stadium_schedule = {
        '20250226': {'teams': ('AFG', 'ENG'), 'stadium': 'Lahore'},
        '20250227': {'teams': ('BAN', 'PAK'), 'stadium': 'Rawalpindi'},
        '20250228': {'teams': ('AFG', 'AUS'), 'stadium': 'Lahore'},
        '20250301': {'teams': ('ENG', 'SA'), 'stadium': 'Karachi'},
        '20250302': {'teams': ('IND', 'NZ'), 'stadium': 'Dubai (DICS)'},
        '20250304': {'teams': (), 'stadium': 'Dubai (DICS)'},
        '20250305': {'teams': (), 'stadium': 'Lahore'},
        '20250309': {'teams': (), 'stadium': 'Lahore'}
    }

    def get_stadium_and_style(date, team1, team2):
        # Get stadium first
        if date == '20250309':
            if team1 == 'IND' or team2 == 'IND':
                stadium = 'Dubai (DICS)'
            else:
                stadium = 'Lahore'
        else:
            stadium = stadium_schedule.get(date, {}).get('stadium', "Stadium not found")
        
        # Get preferred style from ground_details
        if stadium == "Stadium not found":
            preferred_style = 'BAL'
        else:
            # Try to find the stadium in ground_details
            stadium_info = ground_details[ground_details['ground_name'] == stadium]
            if len(stadium_info) > 0:
                preferred_style = stadium_info.iloc[0]['preferred_style']
            else:
                preferred_style = 'BAL'  # Default if stadium not found in details
        
        return stadium, preferred_style

    current_date = datetime.now().strftime('%Y%m%d')
    stadium, style = get_stadium_and_style(current_date, team1, team2)

    return team1, team1_players, team2, team2_players, stadium, style, None

def predict_match_players(df, model, team1, team2, team1_players, team2_players, ground=None, pitch_type=None, batting_first=None):
    """Make predictions for all players in the match"""
    predictions = []
    
    for team, players, opponent in [(team1, team1_players, team2), (team2, team2_players, team1)]:
        for player in players:
            try:
                # Use case-insensitive matching for player names
                player_data = df[df['Player Name'].str.lower() == player.lower()]
                if player_data.empty:
                    print(f"No historical data found for {player}")
                    continue
                
                predicted_points, confidence_interval, details = predict_next_match(
                    model, 
                    player_data,
                    opponent,
                    ground=ground,
                    pitch_type=pitch_type
                )
                
                if predicted_points is not None:
                    predictions.append({
                        'Player': player,
                        'Team': team,
                        'Predicted_Points': predicted_points,
                        'CI_Lower': confidence_interval[0],
                        'CI_Upper': confidence_interval[1],
                        'Player_Type': player_data['Player Type'].iloc[0],
                        'Recent_Form': player_data['fantasy_points'].tail(5).mean(),
                        'Days_Since_Last': details['days_since_last']
                    })
                
            except Exception as e:
                print(f"Error processing {player}: {str(e)}")
    
    if not predictions:
        return None
    
    # Create DataFrame with a unique index
    return pd.DataFrame(predictions).reset_index(drop=True)

def update_cricket_data():
    try:
        start_date = '25 Feb 2025'
        end_date = (datetime.now() - timedelta(days=1)).strftime('%d %b %Y')
        print(f"Checking for new matches from {start_date} to {end_date}")
        
        print("Loading existing data...")
        existing_df = pd.read_csv('processed_final.csv')
        # print(f"Existing data shape: {existing_df.shape}")
        
        # Scrape new data if needed
        dfs = []
        for stats_type in ['batting', 'bowling', 'fielding']:
            print(f"Scraping {stats_type} stats...")
            df = scrape_multiple_pages(start_date, end_date, stats_type)
            if df is not None:
                print(f"{stats_type} data shape: {df.shape}")
                dfs.append(df)

        if dfs:
            print("Processing new data...")
            df1, df2, df3 = dfs
            final_df = clean_data(merge_dfs(df1, df2, df3))
            print(f"New processed data shape: {final_df.shape}")
            
            try:
                print("Combining with existing data...")
                # print(f"Existing DataFrame columns: {existing_df.columns}")
                # print(f"New DataFrame columns: {final_df.columns}")
                
                # Define the required columns
                required_columns = [
                    'player', 'runs', 'mins', 'bf', '4s', '6s', 'sr',
                    'inns_batting', 'overs', 'mdns', 'runs_given', 'wkts', 'econ',
                    'inns_bowling', 'dis', 'ct', 'st', 'ct_wk', 'ct_fi', 'opposition',
                    'ground', 'start_date', 'country', 'runs_clean', 'Credits',
                    'Player Type', 'Player Name', 'Team', 'Bowling Style', 'fantasy_points',
                    'preferred_style', 'opponent_tier'
                ]
                
                # Drop 'Unnamed: 0' if it exists
                if 'Unnamed: 0' in existing_df.columns:
                    existing_df = existing_df.drop('Unnamed: 0', axis=1)
                
                # Ensure both DataFrames have the same columns
                existing_df = existing_df[required_columns]
                final_df = final_df[required_columns]
                
                # Remove duplicates
                existing_df = existing_df.drop_duplicates()
                
                # Concatenate DataFrames
                combined_df = pd.concat([existing_df, final_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['player', 'start_date'])
                combined_df.to_csv('processed_final.csv', index=False)
                print(f"Added {len(final_df)} new records")
            except Exception as e:
                print(f"Error during data combination: {str(e)}")
                raise
        else:
            print("No new matches found")
            
    except Exception as e:
        print(f"Error in update_cricket_data: {str(e)}")
        raise

def get_squad_data():
    """Get squad data with Docker volume mount handling"""
    # Docker mount path
    docker_path = '/app/data/SquadPlayerNames.csv'
    # Fallback path in container
    
    try:
        # Try reading mounted CSV first
        print("here1")
        df = pd.read_csv(docker_path)
        print("Using mounted SquadPlayerNames.csv")
        return df
    except FileNotFoundError:
        # Fallback to packaged CSV
        print("no squad data")

def final_team(predictions):
    # Create a working copy of the DataFrame with a clean index
    df = predictions.copy().reset_index(drop=True)
    working_df = df.copy()
    
    # Selection logic
    selected_players = []
    required_positions = ['WK', 'BAT', 'BOWL', 'ALL']
    required_teams = list(df['Team'].unique())  # Get actual teams from data

    # First, ensure we have at least one player from each position
    for position in required_positions:
        position_players = working_df[working_df['Player_Type'].str.lower() == position.lower()]
        if not position_players.empty:
            selected_player = position_players.nlargest(1, 'Predicted_Points')
            selected_players.append(selected_player.index[0])
            working_df = working_df.drop(selected_player.index[0])

    # Ensure we have at least one player from each team
    for team in required_teams:
        if team not in df.loc[selected_players, 'Team'].values:
            team_players = working_df[working_df['Team'] == team]
            if not team_players.empty:
                selected_player = team_players.nlargest(1, 'Predicted_Points')
                selected_players.append(selected_player.index[0])
                working_df = working_df.drop(selected_player.index[0])

    # Fill remaining slots with highest points players
    remaining_slots = 11 - len(selected_players)
    if remaining_slots > 0:
        remaining_players = working_df.nlargest(remaining_slots, 'Predicted_Points')
        selected_players.extend(remaining_players.index)

    # Get final selected team
    final_team = df.loc[selected_players].copy()

    # Sort players by expected points
    final_team_sorted = final_team.sort_values('Predicted_Points', ascending=False)

    # Select captain and vice-captain
    captain = final_team_sorted.iloc[0]['Player']
    vice_captain = final_team_sorted.iloc[1]['Player']

    # Create output DataFrame
    output_data = {
        'Player_Name': final_team_sorted['Player'],
        'Team': final_team_sorted['Team'],
        'C/VC': ['C' if name == captain else 'VC' if name == vice_captain else 'NA' 
                for name in final_team_sorted['Player']]
    }

    return pd.DataFrame(output_data)

def main():
    try:
        print("1. Starting update_cricket_data...")
        update_cricket_data()
        
        print("2. Loading processed_final.csv...")
        df = pd.read_csv('processed_final.csv')
        print(f"Loaded DataFrame shape: {df.shape}")
        
        print("3. Getting player inputs...")
        team1_name, team1_players, team2_name, team2_players, venue, pitch_type, batting_first = get_player_inputs()
        print(f"Teams: {team1_name} vs {team2_name}")
        
        print("4. Creating features...")
        df = create_player_features(df)
        print(f"DataFrame shape after feature creation: {df.shape}")
        
        print("5. Selecting features...")
        features = select_features(df)
        X = df[features]
        y = df['fantasy_points']
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        print("6. Training model...")
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        ])
        model.fit(X, y)
        print("Model training complete")
        
        print("7. Making predictions...")
        predictions = predict_match_players(
            df,
            model,
            team1=team1_name,
            team2=team2_name,
            team1_players=team1_players,
            team2_players=team2_players,
            ground=venue,
            pitch_type=pitch_type,
            batting_first=batting_first
        )
        
        if predictions is not None:
            print("8. Creating final team...")
            final_team_df = final_team(predictions)
            print(final_team_df)
            final_team_df.to_csv(f'QuestX_output.csv', index=False)
            print("\nPredictions saved to file.")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()