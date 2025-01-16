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

mix_ups = {"ooiwo": ["oiwoo", "ooiwo"], "nsimm22":["nsimm22 "], "rachelrotstein": ["rachrot "]}

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


def extract_leaderboard(uploaded_files):

    all_leaderboards = pd.DataFrame()  # Initialize an empty DataFrame
    for uploaded_file in uploaded_files:
        st.write("processing image...")
        # Save the uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
            # Load the image using OpenCV
            image = cv2.imread(uploaded_file.name)

            # Perform OCR using EasyOCR
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(image)

            # Extract text lines from EasyOCR output
            text_lines = [result[1] for result in results]

            # Create a dictionary to hold the leaderboard data
            chart_data = text_lines

            leaderboard_dict = {}

            loop = 0

            for i in range(len(chart_data)):
                if loop == 0:
                    if any(day in chart_data[i] for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):
                        loop = 1

                        dy = chart_data[i]
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

                                break
                        continue
                
                else:
                    if "(you)" in chart_data[i]:
                        chart_data[i] = chart_data[i].replace('(you)', '')
                        st.session_state["set_user"] = chart_data[i]
                        st.toast("Welcome: " + st.session_state["set_user"])

                    if "Settings" in chart_data[i]:
                        break

                    elif len(chart_data[i]) == 6 and ":" in chart_data[i] and "." in chart_data[i]:
                        chart_data[i] = chart_data[i].replace('.', '')
                        minutes, seconds = chart_data[i].split(':')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[text_lines[i-1]] = total_seconds

                    elif len(chart_data[i]) == 5 and ":" in chart_data[i]:
                        minutes, seconds = chart_data[i].split(':')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[text_lines[i-1]] = total_seconds

                    elif len(chart_data[i]) == 5 and "." in chart_data[i]:
                        minutes, seconds = chart_data[i].split('.')
                        total_seconds = int(minutes) * 60 + int(seconds)
                        leaderboard_dict[text_lines[i-1]] = total_seconds

                    else:
                        print("fail", chart_data[i-1])

            # Convert to a DataFrame and transpose it
            leaderboard_df = pd.DataFrame(leaderboard_dict, index=[datetime_str])

            # Append the current leaderboard to the all_leaderboards DataFrame
            all_leaderboards = pd.concat([all_leaderboards, leaderboard_df])

    return all_leaderboards

def fix_mix_ups(all_leaderboards_post):
    for key in mix_ups.keys():
        for name in mix_ups[key]:
            if name in all_leaderboards_post.columns and key in all_leaderboards_post.columns:
                conflict_mask = ~all_leaderboards_post[name].isna() & ~all_leaderboards_post[key].isna()
                if conflict_mask.any():
                    raise ValueError("Conflict detected: Both 'oiwoo' and 'ooiwoo' have non-NaN values in the same row.")
                all_leaderboards_post[key] = all_leaderboards_post[key].combine_first(all_leaderboards_post[name])
                all_leaderboards_post.drop(columns=[name], inplace=True)

def post_process(all_leaderboards):
    all_leaderboards = all_leaderboards.dropna(axis=1, how='all')

    all_leaderboards_post = all_leaderboards.copy()
        # Post-processing: Drop the 'Datetime' column and handle merging

    all_leaderboards_post = fix_mix_ups(all_leaderboards_post)
    # Remove periods from the index
    #all_leaderboards_post.index = all_leaderboards_post.index.str.replace('.', '')

    # Split at the comma and take the second half
    #all_leaderboards_post.index = all_leaderboards_post.index.str.split(',', n=1).str[1].str.strip()

    # Split at the space to separate month and day-year
    #all_leaderboards_post.index = all_leaderboards_post.index.str.split(' ', n=1)

    # Extract month, day, and year
    #month = all_leaderboards_post.index.str[0]
    #day_year = all_leaderboards_post.index.str[1].str.split(',')

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

def pivot_leaderboard(df):
    try:
        # Pivot the DataFrame to restructure it
        pivoted_df = df.pivot(index='date', columns='username', values='total_seconds')
        return pivoted_df
    except KeyError as e:
        st.write(f"Error pivoting DataFrame: {e}")
        return pd.DataFrame() 

def give_missing_worst_time(df):
    # Fill missing values with the worst time in the row
    df = df.apply(lambda row: row.fillna(row.max()), axis=1)
    return df

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
st.markdown("- If you miss a day, you get the worst time!")
st.markdown("- Upload your Leaderboard Screenshots below to add times to the leaderboard")



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

try:
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

    grace.start_tunnel()
    conn = grace.connect_to_db()
    cursor = grace.conn.cursor()

    st.session_state["conn"] = conn
    st.session_state["cursor"] = cursor

    # Example infinite loop to simulate app behavior
    timeout = 60  # Timeout in seconds


    results = getQuery("""
            SELECT * FROM user_data
             """)

    user_settings = getQuery("""
            SELECT * FROM user_settings
             """)
    
    # Call the function to pivot the DataFrame
    pivoted_results = pivot_leaderboard(results)

    updated_results = fix_mix_ups_results(mix_ups, pivoted_results)

    lb1, lb2 = st.columns(2, gap="medium", border=True)

    lb1.markdown("#### Day to Day")
    lb1.markdown("Note: If the user does not have a result, they are assigned the slowest time")
    lb1.dataframe(pivoted_results)

    fill_missing = give_missing_worst_time(updated_results)

    sum_results = calculate_sum(fill_missing)
    sum_results_today = (sum_results.iloc[-1])
    lb2.markdown("#### Yellow Jersey Leaderboard")
    lb2.write("Todays current standings")
    # Merge sum_results_today with user_settings to get profile images and colors
    leaderboard_with_settings = sum_results_today.reset_index().merge(user_settings, left_on='username', right_on='username', how='left')
    # Rename the current column to 'total_seconds'
    leaderboard_with_settings.rename(columns={leaderboard_with_settings.columns[1]: 'total_seconds'}, inplace=True)

    # Sort by total_seconds
    leaderboard_with_settings = leaderboard_with_settings.sort_values(by='total_seconds')

    # Generate HTML for the leaderboard with rankings
    leaderboard_html = "<div style='flex-direction: column; align-items: left;'>"
    for rank, row in enumerate(leaderboard_with_settings.iterrows(), start=1):
        _, row = row
        username = row['username']
        total_seconds = row['total_seconds']
        url_link = row['url_link']
        color = row['color']

        if pd.isna(color):
            # Generate a random color if no color is provided
            color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))

        if pd.isna(url_link):
            # If no profile image, use a colored circle with the first letter of the username
            first_letter = username[0].upper()
            profile_html = f"<div style='width: 50px; height: 50px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>{first_letter}</div>"
        else:
            # Use the profile image
            profile_html = f"<img src='{url_link}' style='width: 50px; height: 50px; border-radius: 50%;'>" 

        leaderboard_html += f"<div style='display: flex; align-items: center; padding-left: 30px; margin-bottom: 10px;'><div style='margin-right: 10px; font-size: 24px; font-weight: bold;'>{rank}</div><div style='margin-right: 10px;'>{profile_html}</div><div style='padding-left: 10px;'>{username}: {total_seconds} seconds</div></div>"

        leaderboard_html += "<div style='display: flex; width: 100%; flex-direction: column; border-top: 1px solid #ccc; align-items: left; padding-top: 5px; padding-bottom: 5px;'></div>"
    leaderboard_html += "</div>"

    lb2.markdown(leaderboard_html, unsafe_allow_html=True)



    st.markdown("#### Accumulated results")
    fig = px.scatter(sum_results, x=sum_results.index, y=sum_results.columns, labels={'value': 'Cumulative Seconds [s]'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title='Date', yaxis_title='Cumulative Seconds [s]')
    fig.update_xaxes(tickformat='%Y-%m-%d')  # Show only the date part on the x-axis
    st.plotly_chart(fig)

    with st.expander("Upload Data"):

        st.markdown("Upload screenshots of your mini-crossword leaderboard")
        st.markdown("The results table will be extracted from the image and pushed to the database")

        # Add an image input selector for multiple images
        uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

        for uploaded_file in uploaded_files:
            file_size = os.path.getsize(uploaded_file.name) / (1024 * 1024)
            st.write(f"File size of {uploaded_file.name}: {file_size:.2f} MB")

        # Process the uploaded images
        if uploaded_files:
            if len(st.session_state["final_data"]) == 0:
                try:
                    all_leaderboards = extract_leaderboard(uploaded_files)

                    all_leaderboards_post = post_process(all_leaderboards)

                    st.session_state["final_data"] = all_leaderboards_post
                except:
                    st.warning("Failed to Process Image")

            else:
                all_leaderboards_post = st.session_state["final_data"] 

            
            st.write("Dataframe to be inserted. Please confirm data and people.")
            st.dataframe(all_leaderboards_post)

            users = st.multiselect("For privacy, you may deselect users", options=list(all_leaderboards_post.columns), default=list(all_leaderboards_post.columns))

            st.session_state["confirmButton"] = st.button("Confirm Data")

            if st.session_state["confirmButton"] and len(all_leaderboards_post) > 0:

                try:
                    all_leaderboards_post = all_leaderboards_post[users]
                    
                    st.write("uploading...")

                    insert_data(all_leaderboards_post)

                    st.session_state["final_data"] = pd.DataFrame()

                except:
                    st.warning("Failed to upload data")
                    st.session_state["final_data"] = pd.DataFrame()


    
    if st.session_state["set_user"] is not None:
        st.markdown("#### Welcome: " + st.session_state["set_user"])
        st.markdown("Edit your profile below")

        c1,c2 = st.columns(2)

        new_url = c1.text_input("Copy an image Url to change your profile picture!")
        user_color = c2.color_picker("Pick a color for your profile!")
        submit = c1.button("Update Profile")
        if submit:
            if len(new_url) >= 2000:
                st.error("Url too long, please use a shorter one" + str(len(new_url)))
            if len(new_url) > 0 and len(new_url) < 2000:
                try:
                    st.session_state["cursor"].execute(""" 
                    INSERT INTO user_settings (username, url_link)
                    VALUES (%s, %s)
                    ON CONFLICT (username)
                    DO UPDATE SET url_link = EXCLUDED.url_link;
                    """, (st.session_state["set_user"], new_url))
                    st.session_state["conn"].commit()
                    st.success("Profile picture updated!")
                except Exception as e:
                    st.session_state["conn"].rollback()
                    st.error(f"Error updating profile picture: {str(e)}")
                
            if user_color:
                try:
                    st.session_state["cursor"].execute(""" 
                    INSERT INTO user_settings (username, color)
                    VALUES (%s, %s)
                    ON CONFLICT (username)
                    DO UPDATE SET color = EXCLUDED.color;
                    """, (st.session_state["set_user"], str(user_color)))
                    st.session_state["conn"].commit()
                    st.success("Color updated!")
                except Exception as e:
                    st.session_state["conn"].rollback()
                    st.error(f"Error updating color: {str(e)}")
                
                st.markdown(f"<div style='width: 200px; height: 50px; background-color: {user_color};'></div>", unsafe_allow_html=True)



finally:
    # Ensure all resources are cleaned up
    grace.close_resources()
    st.session_state.grace = None