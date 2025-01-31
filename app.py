import streamlit as st
import pandas as pd
import easyocr
import numpy as np
import cv2
import tempfile
from sshtunnel import SSHTunnelForwarder
import time
import psycopg2
import plotly.express as px
import os
from PIL import Image
import io
import base64
from datetime import date, datetime


mix_ups = {"ooiwo": ["oiwoo", "ooiwo"], "nsimm22":["nsimm22 "], "rachelrotstein": ["rachrot ", "rachrot"]}

if "set_user" not in st.session_state:
    st.session_state["set_user"] = None
if "confirmButton" not in st.session_state:
    st.session_state["confirmButton"] = False
if "final_data" not in st.session_state:
    st.session_state["final_data"] = pd.DataFrame()

class GracefulSSHTunnel:
    def __init__(self, ssh_username, ssh_password, ssh_private_key, db_host, db_port, db_name, db_user, db_password):
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_private_key = ssh_private_key
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.tunnel = None
        self.conn = None
        self.temp_key_path = None

        # Write the private key to a temporary file
        with tempfile.NamedTemporaryFile("w", delete=False) as temp_key_file:
            temp_key_file.write(self.ssh_private_key)
            self.temp_key_path = temp_key_file.name

    def start_tunnel(self):
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                self.tunnel = SSHTunnelForwarder(
                    ssh_address_or_host=('ssh.pythonanywhere.com', 22),
                    ssh_username=self.ssh_username,
                    ssh_pkey=self.temp_key_path,
                    ssh_private_key_password=self.ssh_password,
                    remote_bind_address=(self.db_host, self.db_port),
                    local_bind_address=('127.0.0.1', 0)
                )
                self.tunnel.start()

                print(f"SSH Tunnel started on dynamic port: {self.tunnel.local_bind_port}")
                return self.tunnel
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                time.sleep(4)  # Wait for 2 seconds before retrying

        st.warning("Failed to start SSH Tunnel after 5 attempts.")
        self.close_resources()
        raise RuntimeError("Failed to start SSH Tunnel after 5 attempts.")

    def connect_to_db(self):
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                if not self.tunnel or not self.tunnel.is_active:
                    raise RuntimeError("SSH tunnel is not active. Start the tunnel before connecting to the database.")
                
                self.conn = psycopg2.connect(
                    host='127.0.0.1',  # Local address of the tunnel
                    port=self.tunnel.local_bind_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
                print("Database connection established.")
                return self.conn
            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} to connect to the database failed: {e}")
                time.sleep(4)  # Wait for 2 seconds before retrying

        st.warning("Failed to connect to the database after 5 attempts.")
        self.close_resources()
        raise RuntimeError("Failed to connect to the database after 5 attempts.")

    def close_resources(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.stop()
            print("SSH Tunnel closed.")
        if self.temp_key_path and os.path.exists(self.temp_key_path):
            os.remove(self.temp_key_path)
            print("Temporary private key file removed.")

def run_ocr(uploaded_file):
        
    # Use a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file.flush()  # Ensure the file is written
        
        # Reset the file pointer to the beginning of the file
        temp_file.seek(0)

        # Read the image data from the temporary file
        image_data = np.frombuffer(temp_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)  # Decode the image

        if image is None:
            st.warning("Failed to decode image. Please check the uploaded file format.")
            return  # Skip to the next file if decoding fails

        # Perform OCR using EasyOCR
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(image)

            # Extract text lines from EasyOCR output
            text_lines = [result[1] for result in results]

            # Create a dictionary to hold the leaderboard data
            chart_data = text_lines

            if len(text_lines) < 1:
                st.write("No text detected in the image. Please try again.")

            return chart_data

        except Exception as e:
            st.warning(f"Error during OCR processing: {str(e)}")
            return []


def find_crossword_day(chart_data, index):
    if any(day in chart_data[index] for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):

        dy = chart_data[index]
        # Remove the day from the string
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            if day in dy:
                datetime_str_new = dy.replace(day, '').strip()
        
                # Extract and remove the month short form
                month_short_forms = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                month = next((month for month in month_short_forms if month in datetime_str_new), None)
                if month:
                    datetime_str_new = datetime_str_new.replace(month, '').strip()
                
                # Extract and remove the year
                year_start_idx = datetime_str_new.find("202")
                if year_start_idx != -1:
                    year = datetime_str_new[year_start_idx:year_start_idx + 4]
                    datetime_str_new = datetime_str_new.replace(year, '').strip()
        
                # The remaining number is the date
                tdate = ''.join(filter(str.isdigit, datetime_str_new))

                datetime_str = f"{month} {tdate}, {year}"

                return datetime_str
    return False

def extract_leaderboard(uploaded_files):

    all_leaderboards = pd.DataFrame()  # Initialize an empty DataFrame for final leaderboard

    for uploaded_file in uploaded_files:

        with st.spinner("Processing Image..."):
            chart_data = run_ocr(uploaded_file)
        if chart_data == None or len(chart_data) == 0:
            return pd.DataFrame()
        elif len(chart_data) > 0:
            leaderboard_dict = {}
            
            #First find the date, then skip to the time / user finder
            loop = 0

            for i in range(len(chart_data)):
                if loop == 0:
                    crossword_day = find_crossword_day(chart_data, i)
                    if crossword_day == False:
                        continue
                    else:
                        #Date found, skip to time:user finder
                        loop = 1
                    
                #time:user finder 
                else:
                    #Allows current user to be identified, because only the logged in user on NYT will have (you) in their name
                    if "(you)" in chart_data[i]:
                        #Remove the (you), so the username is normalized
                        chart_data[i] = chart_data[i].replace('(you)', '')
                        #set the current user in session state
                        st.session_state["set_user"] = chart_data[i]
                        st.toast("Welcome: " + st.session_state["set_user"])

                    #Settings is always after the table data
                    if "Settings" in chart_data[i]:
                        break

                    #If the len is 6, it is usally a time but has mistaken the : for a :.
                    elif len(chart_data[i]) == 6 and ":" in chart_data[i] and "." in chart_data[i]:
                        chart_data[i] = chart_data[i].replace('.', '')
                        minutes, seconds = chart_data[i].split(':')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[chart_data[i-1]] = total_seconds

                    #If len is 5, it usually in th form 00:00
                    elif len(chart_data[i]) == 5 and ":" in chart_data[i]:
                        minutes, seconds = chart_data[i].split(':')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[chart_data[i-1]] = total_seconds

                    #If len is 5, it usually in th form 00:00 but sometimes the ocr mistakes it for a 00.00
                    elif len(chart_data[i]) == 5 and "." in chart_data[i]:
                        minutes, seconds = chart_data[i].split('.')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[chart_data[i-1]] = total_seconds

                    #if none of those declare failure and print the problem child - could be more ways for the ocr to fuck up
                    else:
                        print("Failed to process, not a time", chart_data[i])
                        continue

            # Convert to a DataFrame and transpose it
            leaderboard_df = pd.DataFrame(leaderboard_dict, index=[crossword_day])

            # Append the current leaderboard to the all_leaderboards DataFrame
            all_leaderboards = pd.concat([all_leaderboards, leaderboard_df])

    return all_leaderboards

#OCR sometimes messes up usernames, keep a list of the ones that occur and have them pushed together
def fix_mix_ups(all_leaderboards_post):
    for key in mix_ups.keys():
        for name in mix_ups[key]:
            if name in all_leaderboards_post.columns and key in all_leaderboards_post.columns:
                conflict_mask = ~all_leaderboards_post[name].isna() & ~all_leaderboards_post[key].isna()
                if conflict_mask.any():
                    raise ValueError(f"Conflict detected: Both problem: '{name}' and normal: '{key}' have non-NaN values in the same row.")
                all_leaderboards_post[key] = all_leaderboards_post[key].combine_first(all_leaderboards_post[name])
                all_leaderboards_post.drop(columns=[name], inplace=True)
    return all_leaderboards_post

def post_process(all_leaderboards):
    all_leaderboards = all_leaderboards.dropna(axis=1, how='all')

    all_leaderboards_post = all_leaderboards.copy()
    # Post-processing: Drop the 'Datetime' column and handle merging

    all_leaderboards_post = fix_mix_ups(all_leaderboards_post)

    # Combine into a datetime object
    all_leaderboards_post.index = pd.to_datetime(all_leaderboards_post.index, errors="coerce", format='%b %d, %Y')

    return all_leaderboards_post

def insert_data(all_leaderboards_post):

    try:
        # Melt the dataframe to convert from wide to long format
        df_melted = all_leaderboards_post.reset_index().melt(
            id_vars=['index'],
            var_name='username',
            value_name='total_seconds'
        )
        
        # Rename the index column to date
        df_melted = df_melted.rename(columns={'index': 'date'})
        
        # Drop any rows where total_seconds is NaN
        df_melted = df_melted.dropna(subset=['total_seconds'])
        
        # Insert each row into the database
        for _, row in df_melted.iterrows():
            st.session_state["cursor"].execute(""" 
                INSERT INTO user_data (date, username, total_seconds)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, username) 
                DO UPDATE SET total_seconds = EXCLUDED.total_seconds
                WHERE user_data.total_seconds = EXCLUDED.total_seconds;
            """, (row['date'], row['username'], row['total_seconds']))
        
        # Commit the transaction
        st.session_state["conn"].commit()
        st.success("Data successfully inserted into database!")
        
    except Exception as e:
        st.session_state["conn"].rollback()
        st.error(f"Error inserting data: {str(e)}")

#some
def fix_mix_ups_results(mix_ups, all_leaderboards_post):
    for key in mix_ups.keys():
        for name in mix_ups[key]:
            if name in all_leaderboards_post.columns and key in all_leaderboards_post.columns:
                conflict_mask = ~all_leaderboards_post[name].isna() & ~all_leaderboards_post[key].isna() & (all_leaderboards_post[name] != all_leaderboards_post[key])
                if conflict_mask.any():
                    raise ValueError(f"Conflict detected: Both '{key}' and '{name}' have different non-NaN values in the same row.")
                all_leaderboards_post[key] = all_leaderboards_post[key].combine_first(all_leaderboards_post[name])
                all_leaderboards_post.drop(columns=[name], inplace=True)
            elif name in all_leaderboards_post.columns:
                all_leaderboards_post.rename(columns={name: key}, inplace=True)

    return all_leaderboards_post

#Pivot from database list style to more natural
def pivot_leaderboard(df):
    try:
        # Pivot the DataFrame to restructure it
        pivoted_df = df.pivot(index='date', columns='username', values='total_seconds')
        return pivoted_df
    except KeyError as e:
        st.write(f"Error pivoting DataFrame: {e}")
        return pd.DataFrame() 

def give_missing_worst_time(df, num_skips):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Fill missing values with the worst time in the row
    df_copy = df_copy.apply(lambda row: row.fillna(row.max()), axis=1)

    # Check the number of entries for each user
    for user in df.columns:
        if user != 'date':  # Skip the 'date' column
            # Get the user's times
            user_times = df_copy[user]  # Keep all values, including NaN
            
            # Only change the largest times to zero if there are more than 5 entries
            if len(user_times) > 5:
                # Identify the largest times, ignoring NaN values
                largest_times = user_times.nlargest(num_skips).index.tolist()
                # Set the largest times to zero
                df_copy.loc[largest_times, user] = 0

    return df_copy

def calculate_sum(df):
    # for each player, update the dataframe so it has the username plus their total time at every day, where the time is the sum of all previous days
    for column in df.columns:
        df[column] = df[column].cumsum()
    return df


# Set page configuration
st.set_page_config(
    page_title="NYT Mini Battle",
    page_icon="🃏",
    layout="wide"
)

# Track user activity
if 'last_active' not in st.session_state:
    st.session_state.last_active = time.time()

# Update activity timestamp
st.session_state.last_active = time.time()

# Add centered markdown with bold title
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Palmerston & Friends - NYT Mini Leaderboard</h1>", unsafe_allow_html=True)
st.markdown("#### Rules")
st.markdown("- Tour de France Style - Best accumulated time at the end of the month wins!")
st.markdown("- If you miss a day, you are assigned the groups's worst time, with your 3 worst times getting dropped")
st.markdown("- Upload your Leaderboard Screenshots below to add times to the leaderboard")

def print_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)  # Count the depth of the current directory
        indent = ' ' * 4 * (level)  # Indentation for tree structure
        print(f"{indent}{os.path.basename(root)}/")  # Print the directory name
        subindent = ' ' * 4 * (level + 1)  # Indentation for files
        for f in files:
            print(f"{subindent}{f}")  # Print the file name

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_leaderboard_html(leaderboard_with_settings):
    # Generate HTML for the leaderboard with rankings
    leaderboard_html = "<div style='flex-direction: column; align-items: left;'>"
    for rank, row in enumerate(leaderboard_with_settings.iterrows(), start=1):
        _, row = row
        username = row['username']
        total_seconds = row['total_seconds']

        # Construct the path to the user's profile picture
        user_photo_path = f"user_photos/{username.strip()}/profile_picture.jpg"  # Adjust the extension as needed

        # Check if the profile image exists
        if os.path.exists(user_photo_path):
            # Print the path for debugging
            print(f"Profile image path: {user_photo_path}")
            
            # Get the base64 encoded image
            base64_image = get_base64_image(user_photo_path)
    
            # Use the profile image in HTML with cropping
            profile_html = f"<img src='data:image/jpeg;base64,{base64_image}' style='width: 50px; height: 50px; border-radius: 50%; object-fit: cover;'>"
        else:
            # Generate a random color for each user
            color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))
            # Use a colored circle with the first letter of the username
            first_letter = username[0].upper()
            profile_html = f"<div style='width: 50px; height: 50px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>{first_letter}</div>"

        leaderboard_html += f"<div style='display: flex; align-items: center; padding-left: 30px; margin-bottom: 10px;'><div style='margin-right: 10px; font-size: 24px; font-weight: bold;'>{rank}</div><div style='margin-right: 10px;'>{profile_html}</div><div style='padding-left: 10px;'>{username}: {int(total_seconds)} seconds</div></div>"

        leaderboard_html += "<div style='display: flex; width: 100%; flex-direction: column; border-top: 1px solid #ccc; align-items: left; padding-top: 5px; padding-bottom: 5px;'></div>"
    leaderboard_html += "</div>"
    return leaderboard_html

def create_plot(sum_results):
    
    fig = px.scatter(sum_results, x=sum_results.index, y=sum_results.columns, labels={'value': 'Cumulative Seconds [s]'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Seconds [s]')
    fig.update_xaxes(tickformat='%Y-%m-%d')  # Show only the date part on the x-axis

    return  fig

def count_nones(pivoted):
    # Count None (NaN) values in a specific column
    skip_dict = {}
    for col in pivoted.columns:
        none_count = pivoted[col].isna().sum()
        skip_dict[col] = [none_count]
    
    skip_df = pd.DataFrame.from_dict(skip_dict).T
    
    # Rename the final DataFrame column to "Missed Days"
    skip_df.columns = ["Missed Days"]
    
    return skip_df

#General Query funciton, returns a dataframe. Use this instead of pd.read_sql
def getQuery(query, params=None):
    if params is None:
        params = []
    try:
        st.session_state["cursor"].execute(query, params)
        Data = pd.DataFrame.from_records(
            st.session_state["cursor"].fetchall(), 
            columns=[col.name for col in st.session_state["cursor"].description]
        )
        return Data
    except psycopg2.Error as e:
        st.write(f"Error executing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def apply_penalties(data, year, month, day, quantity, user):
    try:
        data.loc[data.index==date(year,month,day), user] = quantity
    except:
        pass
    return data

def apply_penalties_batch(data):
    data = apply_penalties(data, 2025, 1, 18, 60, 'rachelrotstein')
    data = apply_penalties(data, 2025, 1, 16, -20, 'hankthetank14')
    return data

def get_current_month(full_results, current_month_str):
    # Convert the month string to a month number
    month_number = datetime.strptime(current_month_str, "%B").month

    # Convert index to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(full_results.index):
        try:
            full_results.index = pd.to_datetime(full_results.index, errors='raise')  # Raise an error if conversion fails
        except Exception as e:
            st.error(f"Error converting index to datetime: {str(e)}")
    
    # Filter results for the current month and year
    months_results = full_results[full_results.index.month == month_number]

    return months_results

def process_monthly_results(updated_results, num_skips, months):
    # List of all month names
    all_months_results = []  # Initialize a list to hold results for all months

    for month in months:
        # Get results for the current month
        month_results = get_current_month(updated_results, current_month_str=month)
        
        if not month_results.empty:  # Check if there is data for the month
            # Apply the give_missing_worst_time function
            results_filled = give_missing_worst_time(month_results, num_skips)
            # Append the processed results to the all_months_results list
            all_months_results.append(results_filled)

    # Concatenate all processed results into a single DataFrame
    return pd.concat(all_months_results, axis=0) if all_months_results else pd.DataFrame()

def format_month_results(month_results):
    # Create a copy of the original DataFrame to avoid modifying it directly
    formatted_results = month_results.copy()

    # Ensure the 'date' column is in datetime format and format it to show only the date
    if 'date' in formatted_results.columns:
        formatted_results.index = pd.to_datetime(formatted_results.index, errors='coerce')  # Convert to datetime if not already
        formatted_results.index = formatted_results.index.dt.strftime('%Y-%m-%d')  # Format to YYYY-MM-DD as string

    # Convert all other columns from total seconds to mm:ss format
    for col in formatted_results.columns:
        if col != 'date':  # Skip the 'date' column
            formatted_results[col] = formatted_results[col].apply(
                lambda x: (
                    f"{int(x // 60):02}:{int(x % 60):02}" 
                    if pd.notna(x) and isinstance(x, (int, float)) and x >= 0 
                    else "00:00"
                )
            )

    # Sort the DataFrame by 'date' before applying styles
    formatted_results = formatted_results.sort_values(by="date", ascending=False)

    # Function to apply styles
    def highlight_best_worst(s):
        # Check if there is only one entry
        if len(s) == 1:
            return ['background-color: green' for _ in s]  # Style as best time (green)

        best_time = min(s[s != "00:00"])  # Find the best time (excluding zeros)
        worst_time = max(s[s != "00:00"])  # Find the worst time (excluding zeros)
        return [
            'background-color: yellow' if x == "00:00" else
            'background-color: green' if x == best_time else
            'background-color: red' if x == worst_time else
            ''  # No style
            for x in s
        ]

    # Apply the styling to all columns except 'date'
    styled_results = formatted_results.style.apply(highlight_best_worst, subset=formatted_results.columns[0:])  # Exclude 'date' column

    return styled_results

try:
    with st.spinner('Loading...'):
        grace = GracefulSSHTunnel(
            ssh_username=st.secrets["ssh"]["username_ssh"],
            ssh_password=st.secrets["ssh"].get("private_key_passphrase", None),
            ssh_private_key=st.secrets["ssh"]["private_key_ssh"],
            db_name=st.secrets["postgres"]["database_post"],
            db_host=st.secrets["postgres"]["hostname"],
            db_port=st.secrets["postgres"]["port"],
            db_user=st.secrets["postgres"]["username_post"],
            db_password=st.secrets["postgres"]["password_post"],
        )
        
        #Start database connection
        grace.start_tunnel()
        conn = grace.connect_to_db()
        cursor = grace.conn.cursor()

        st.session_state["conn"] = conn
        st.session_state["cursor"] = cursor

        timeout = 60  # Timeout in seconds, if the databsae is connected for 60s, remove connection to avoid overstimulating db
        num_skips = 3 #Set number of skips per month
        #Start homepage functionality and display
        # Create a dropdown for selecting the month
        current_month = date.today().strftime("%B")  # Get the current month name
        months = [date(2000, i, 1).strftime("%B") for i in range(1, 13)]  # List of all month names

        #Get results and process them to be displayed
        results = getQuery("SELECT * FROM user_data")
        pivoted_results = pivot_leaderboard(results)
        updated_results = fix_mix_ups_results(mix_ups, pivoted_results)

        # Call the function to process monthly results
        final_results = process_monthly_results(updated_results, num_skips, months)

        selected_month = st.selectbox("Select Month", months, index=months.index(current_month))

        results_filled = apply_penalties_batch(final_results)

        #get user settings (profile picture)
        user_settings = getQuery("SELECT * FROM user_settings")
        
    # Display
    lb1, lb2 = st.columns(2, gap="medium", border=True)

    #Column 1, Day to Day results
    lb1.markdown(f"#### Day to Day - {selected_month}")
    lb1.markdown("- If the user does not have a result, they are assigned the slowest time")
    lb1.markdown("- The users 3 worst times are dropped")
    #lb1.info('Admin Note: A penalty of 60s was applied to `rachelrotstein` on `2025-01-18` for bullying')

    lb2.markdown(f"#### Yellow Jersey Leaderboard - {selected_month}")
    lb2.write("Todays current standings")

    #Create line plot
    st.markdown("#### Accumulated results")
    pl1, pl2 = st.columns(2, gap="small", border=True)

    st.markdown("### ")
    fc1, fc2, fc3, fc4 = st.columns(4, gap="small", border=True)

    month_results = get_current_month(results_filled, current_month_str=selected_month)

    if len(month_results) > 0:
        # Format the month results for display
        formatted_month_results = format_month_results(month_results)

        # Display the formatted DataFrame in Streamlit
        lb1.dataframe(formatted_month_results, use_container_width=True)
    
        #Column 2, leaderboard
        sum_results = calculate_sum(month_results)

        sum_results_today = (sum_results.iloc[-1])

        # Merge sum_results_today with user_settings to get profile images and colors
        leaderboard_with_settings = sum_results_today.reset_index().merge(user_settings, left_on='username', right_on='username', how='left')
        # Rename the current column to 'total_seconds'
        leaderboard_with_settings.rename(columns={leaderboard_with_settings.columns[1]: 'total_seconds'}, inplace=True)

        # Sort by total_seconds
        leaderboard_with_settings = leaderboard_with_settings.sort_values(by='total_seconds')

        #Generate HMTL for leaderboard
        leaderboard_html = generate_leaderboard_html(leaderboard_with_settings)
        lb2.markdown(leaderboard_html, unsafe_allow_html=True)

        pl1.markdown(f"#### {current_month} Tracker")
        fig = create_plot(sum_results)
        pl1.plotly_chart(fig, key="month")

    else: 
        lb1.warning(f"No Data for {selected_month}")
        lb2.warning(f"No Data for {selected_month}")
        pl1.warning(f"No Data for {selected_month}")
    

    pl2.markdown(f"#### Full Year Tracker")
    sum_results_full = calculate_sum(results_filled)
    fig_full = create_plot(sum_results_full)
    pl2.plotly_chart(fig_full, key="year")


    #3 best times
    fc1.markdown("#### Fastest Times:")
    # Ensure the 'date' column is in datetime format
    results["date"] = pd.to_datetime(results["date"], errors='coerce')
    # Add day of the week column to results
    results["Day of the Week"] = results["date"].dt.day_name()

    # Format the 'date' column to show only the date (YYYY-MM-DD)
    results["date"] = results["date"].dt.strftime('%Y-%m-%d')

    results = results.rename(columns={"total_seconds": "Total Seconds"})
    results = results[["date", "Day of the Week", "username", "Total Seconds"]]

    fc1.dataframe(results.sort_values(by="Total Seconds").head(), use_container_width=True, hide_index=True)

    #3 worst times
    fc2.markdown("#### Slowest Times:")
    # Ensure the 'date' column is in datetime format
    results["date"] = pd.to_datetime(results["date"], errors='coerce')
    # Add day of the week column to results
    results["Day of the Week"] = results["date"].dt.day_name()

    results["date"] = results["date"].dt.strftime('%Y-%m-%d')

    results = results.rename(columns={"total_seconds": "Total Seconds"})
    results = results[["date", "Day of the Week", "username", "Total Seconds"]]

    # Display the DataFrame without the index
    fc2.dataframe(results.sort_values(by="Total Seconds", ascending=False).head(), use_container_width=True, hide_index=True)

    none_count_df = count_nones(updated_results)

    #try hards
    fc3.markdown("#### Biggest Try Hards:")
    fc3.dataframe(none_count_df.sort_values(by=none_count_df.columns[0]).head(), use_container_width=True)

    #deliquents
    fc4.markdown("#### Top Delinquents:")
    fc4.dataframe(none_count_df.sort_values(by=none_count_df.columns[0], ascending=False).head(), use_container_width=True)


    #End display, start upload functionality
    with st.expander("Upload Data"):

        st.markdown("Upload screenshots of your mini-crossword leaderboard")
        st.markdown("The results table will be extracted from the image and pushed to the database")

        # Clear the session state for the file uploader if needed
        if "uploaded_files" in st.session_state:
            del st.session_state["uploaded_files"]

        # File uploader
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Initialize a list to hold resized images
        resized_images = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Check file size
                    file_size = uploaded_file.size / (1024 * 1024)
                    if file_size > 0.2:
                        st.write(f"File size of {uploaded_file.name} is {file_size:.2f} MB, too large...")
                        
                        # Resize the image
                        image = Image.open(uploaded_file)

                        # Define the new size (you can adjust this as needed)
                        new_size = (int(image.width * 0.3), int(image.height * 0.3))  # Resize to 30% of original
                        image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS for high-quality downsampling

                        # Save the resized image to a bytes buffer with reduced quality
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG', quality=85)  # Save as JPEG with quality 85
                        img_byte_arr.seek(0)  # Move to the beginning of the BytesIO buffer

                        # Calculate the size of the resized image
                        reduced_size = img_byte_arr.tell() / (1024 * 1024)  # Size in MB
                        st.write(f"Reduced size of {uploaded_file.name} is {reduced_size:.2f} MB.")

                        # Append the resized image to the list
                        resized_images.append(img_byte_arr)

                    else:
                        # If the file is not resized, append the original uploaded file
                        resized_images.append(uploaded_file)


                except Exception as e:
                    st.warning(f"Error processing file {uploaded_file.name}: {str(e)}")
                    break

            # Pass resized_images to the extract_leaderboard function
            # Clear the session state for the file uploader if needed
            if "uploaded_files" in st.session_state:
                del st.session_state["uploaded_files"]
            
            if len(st.session_state["final_data"]) == 0:

                all_leaderboards = extract_leaderboard(resized_images)

                all_leaderboards_post = post_process(all_leaderboards)

                st.session_state["final_data"] = all_leaderboards_post
            
            else:
                all_leaderboards_post = st.session_state["final_data"]

            if len(all_leaderboards_post) > 0:
                st.write("Dataframe to be inserted. Please confirm data and people.")
                st.dataframe(all_leaderboards_post)

                users = st.multiselect("For privacy, you may deselect users", options=list(all_leaderboards_post.columns), default=list(all_leaderboards_post.columns))

                st.session_state["confirmButton"] = st.button("Confirm Data")

                if st.session_state["confirmButton"] and len(all_leaderboards_post) > 0:

                    try:
                        all_leaderboards_post = all_leaderboards_post[users]
                        
                        with st.spinner("uploading..."):
                            insert_data(all_leaderboards_post)

                        st.session_state["final_data"] = pd.DataFrame()

                    except Exception as e:
                        st.warning(f"Failed to Push to DB: {str(e)}")
                        st.session_state["final_data"] = pd.DataFrame()


    
    #If user uploads a file with (you), it logs them in
    if st.session_state["set_user"] is not None:
        st.markdown("#### Welcome: " + st.session_state["set_user"])
        st.markdown("Edit your profile below")

        c1,c2 = st.columns(2)

        uploaded_file = c1.file_uploader("Upload an image to change your profile picture!", type=["png", "jpg", "jpeg"])
        submit = c1.button("Update Profile")
        if submit:
            if uploaded_file is not None:
                try:
                    # Create directory if it doesn't exist
                    user_dir = f"user_photos/{st.session_state['set_user'].strip()}"  # Strip whitespace
                    os.makedirs(user_dir, exist_ok=True)

                    # Save the uploaded image to the specified directory
                    with open(f"{user_dir}/profile_picture{os.path.splitext(uploaded_file.name)[1]}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Profile picture saved!")
                except Exception as e:
                    st.error(f"Error saving profile picture: {str(e)}")
            else:
                st.error("Please upload an image file.")


# Ensure all resources are cleaned up
finally:
    grace.close_resources()
    st.session_state.grace = None