import pandas as pd

CAPACITY_PATH = "/Users/svaradeshmukh/WRugby_Match_Prediction/venue_capacities.csv"
matches = "/Users/svaradeshmukh/WRugby_Match_Prediction/data/matches_with_capacity.csv"

capacity = pd.read_csv(CAPACITY_PATH)
match = pd.read_csv(matches)

venues = set(match['venue'].tolist())
existing_cap = capacity['venue'].tolist()


missing = [item for item in venues 
           if item not in existing_cap]

print(missing)
