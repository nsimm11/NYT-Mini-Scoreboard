import pandas as pd
import cv2
import easyocr
import os
import numpy as np

#In the current directory, find all relative filepaths of images
def get_image_paths():
    image_paths = []
    for root, dirs, files in os.walk("."):
        print(root, dirs, files)
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths


def extract_leaderboard(image_path, all_leaderboards):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Perform OCR using EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    # Extract text lines from EasyOCR output
    text_lines = [result[1] for result in results]

    print(text_lines[4:3+6*2-1])

    datetime_str = text_lines[3]

    # Create a dictionary to hold the leaderboard data
    leaderboard_dict = {"Datetime": datetime_str}

    # Populate the dictionary with usernames as keys and their values
    for i in range(4, 2+6*2+1, 2):
        if "(you)" in text_lines[i]:
            text_lines[i] = text_lines[i].replace(" (you)", "")
        time_str = text_lines[i+1]
        if "." in time_str:
            minutes, seconds = map(int, time_str.split('.'))
        elif ":" in time_str:
            minutes, seconds = map(int, time_str.split(':'))
        else:
            print("Error: Time format not recognized", time_str)
            minutes, seconds = np.nan, np.nan
        time_value = pd.to_timedelta(minutes, unit='m') + pd.to_timedelta(seconds, unit='s')
        leaderboard_dict[text_lines[i]] = time_value

    # Convert to a DataFrame and transpose it
    leaderboard_df = pd.DataFrame(leaderboard_dict, index=[datetime_str])

    # Append the current leaderboard to the all_leaderboards DataFrame
    all_leaderboards = pd.concat([all_leaderboards, leaderboard_df])

    return all_leaderboards

# Start loop
image_paths = get_image_paths()
all_leaderboards = pd.DataFrame()

for image_path in image_paths:
    all_leaderboards = extract_leaderboard(image_path, all_leaderboards)

# Display the combined DataFrame
all_leaderboards.drop(columns=["Datetime"], inplace=True)
print(all_leaderboards)
# Merge columns 'oiwoo' and 'ooiwoo'
if 'oiwoo' in all_leaderboards.columns and 'ooiwoo' in all_leaderboards.columns:
    # Check if there are any non-NaN values in both columns
    conflict_mask = ~all_leaderboards['oiwoo'].isna() & ~all_leaderboards['ooiwoo'].isna()
    if conflict_mask.any():
        raise ValueError("Conflict detected: Both 'oiwoo' and 'ooiwoo' have non-NaN values in the same row.")
    
    # Merge the columns, prioritizing non-NaN values from 'oiwoo'
    all_leaderboards['ooiwoo'] = all_leaderboards['ooiwoo'].combine_first(all_leaderboards['oiwoo'])
    all_leaderboards.drop(columns=['oiwoo'], inplace=True)

print(all_leaderboards.sort_index(axis=1))
