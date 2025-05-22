import os
import csv

# File paths
filename = 'position data/apophis_pos.txt'
output_csv = 'position data/apophis_data.csv'

# Line indices to extract (adjust as needed)
start_line = 87  # line where data begins (after $$SOE)
end_line = 9389  # line where data ends (before $$EOE)

def parse_jpl_horizons_data(lines):
    """Parse JPL Horizons format data into structured rows"""
    parsed_data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('$$EOE'):
            continue
            
        # Split the line into components (handling variable whitespace)
        parts = line.split()
        
        # Basic validation
        if len(parts) < 17:
            print(f"Skipping malformed line: {line}")
            continue
            
        # Extract date/time
        date_ut = ' '.join(parts[:2])
        
        # Extract RA (hours, minutes, seconds)
        ra_h, ra_m, ra_s = parts[2:5]
        
        # Extract Dec (sign is part of degrees)
        dec_sign = '-' if parts[5].startswith('-') else '+'
        dec_d, dec_m, dec_s = parts[5].lstrip('+-'), parts[6], parts[7].rstrip(',')
        
        # Extract remaining fields
        remaining = parts[8:]
        
        # Handle special case where S-O-T might contain spaces
        if len(remaining) > 10:
            # Join any extra fields that might belong to S-O-T
            remaining[2] = ' '.join(remaining[2:2+(len(remaining)-10)])
            del remaining[3:3+(len(remaining)-10)]
        
        parsed_data.append([
            date_ut, ra_h, ra_m, ra_s, dec_sign, dec_d, dec_m, dec_s
        ] + remaining)
    
    return parsed_data

# Check if file exists
if os.path.exists(filename):
    with open(filename, 'r') as f:
        # Read only the data portion (between $$SOE and $$EOE)
        lines = []
        in_data = False
        for line in f:
            if line.startswith('$$SOE'):
                in_data = True
                continue
            if line.startswith('$$EOE'):
                break
            if in_data and line.strip():
                lines.append(line)
    
    # Parse the data
    parsed_data = parse_jpl_horizons_data(lines)
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        
        # Header with RelTime as the first column
        headers = [
            "RelTime", "Date_(UT)_HR:MN", "RA_h", "RA_m", "RA_s", 
            "Dec_sign", "Dec_d", "Dec_m", "Dec_s",
            "APmag", "S-brt", "delta", "deldot", "S-O-T", "S-T-O",
            "Sky_motion", "Sky_mot_PA", "RelVel-ANG", "Lun_Sky_Brt", "sky_SNR"
        ]
        writer.writerow(headers)
        
        # Add RelTime index and write rows
        for i, row in enumerate(parsed_data):
            if len(row) >= 17:
                writer.writerow([i] + row[:17])  # Only take first 17 columns if more exist
            else:
                print(f"Skipping incomplete row {i}: {row}")
                
    print(f"Successfully extracted {len(parsed_data)} rows to {output_csv}")
else:
    print(f"File not found: {filename}")