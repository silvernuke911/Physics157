import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the CSV with proper parsing
def load_and_clean_data(filepath):
    # Read the CSV while handling the merged delta/deldot column
    df = pd.read_csv(filepath, index_col=False)
    
    # Split the merged delta/deldot column
    delta_split = df['delta'].str.split(' ', expand=True)
    df['delta'] = delta_split[0]  # First part is delta
    df['deldot'] = delta_split[1] if len(delta_split.columns) > 1 else np.nan  # Second part is deldot
    
    # Convert columns to numeric where appropriate
    numeric_cols = ['RA_h', 'RA_m', 'RA_s', 'Dec_d', 'Dec_m', 'Dec_s', 
                   'APmag', 'S-brt', 'delta', 'deldot', 'Sky_motion', 
                   'Sky_mot_PA', 'RelVel-ANG']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load and clean the data
df = load_and_clean_data('position data/apophis_data.csv')
print(df.head())

# Function to convert RA from h/m/s to radians
def ra_to_rad(h, m, s):
    ra_hours = h + m / 60 + s / 3600
    return math.radians(ra_hours * 15)  # 15 degrees per hour

# Function to convert Dec from sign/d/m/s to radians
def dec_to_rad(sign, d, m, s):
    dec_degrees = d + m / 60 + s / 3600
    if isinstance(sign, str) and sign.strip().startswith('-'):
        dec_degrees *= -1
    return math.radians(dec_degrees)

# Clean and convert RA, Dec, and delta
x_list, y_list, z_list = [], [], []

for idx, row in df.iterrows():
    try:
        if pd.isna(row['delta']) or str(row['delta']).strip().lower() == 'n.a.':
            raise ValueError("delta is 'n.a.' or missing")

        ra_rad = ra_to_rad(row['RA_h'], row['RA_m'], row['RA_s'])
        dec_rad = dec_to_rad(str(row['Dec_sign']), row['Dec_d'], row['Dec_m'], row['Dec_s'])
        r = row['delta']

        x = r * math.cos(dec_rad) * math.cos(ra_rad)
        y = r * math.cos(dec_rad) * math.sin(ra_rad)
        z = r * math.sin(dec_rad)

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    except Exception as e:
        print(f"Skipping row {idx} due to error: {e}")
        x_list.append(np.nan)
        y_list.append(np.nan)
        z_list.append(np.nan)

# Add to new DataFrame
df_xyz = pd.DataFrame({
    'RelTime': df['RelTime'],
    'UT': df['Date_(UT)_HR:MN'],
    'x': x_list,
    'y': y_list,
    'z': z_list
})
print(df_xyz.head())

# Save the computed cartesian coordinates
df_xyz.to_csv('position data/apophis_cartesian.csv', index=False)
print("Saved: position data/apophis_cartesian.csv")

# Plot the first 1000 values of x, y, z
plt.figure(figsize=(12, 6))
plt.plot(df_xyz['RelTime'][:1000], df_xyz['x'][:1000], label='x', marker='.', linewidth=0.8)
plt.plot(df_xyz['RelTime'][:1000], df_xyz['y'][:1000], label='y', marker='.', linewidth=0.8)
plt.plot(df_xyz['RelTime'][:1000], df_xyz['z'][:1000], label='z', marker='.', linewidth=0.8)

plt.xlabel('RelTime')
plt.ylabel('Distance (AU)')
plt.title('Apophis Cartesian Coordinates (First 1000 Points)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()