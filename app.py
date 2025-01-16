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

mix_ups = {"ooiwo": ["oiwoo"]}

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
                    if "Monday" in chart_data[i] or "Tuesday" in chart_data[i] or "Wednesday" in chart_data[i] or "Thursday" in chart_data[i] or "Friday" in chart_data[i] or "Saturday" in chart_data[i] or "Sunday" in chart_data[i]:
                        loop = 1
                        datetime_str = chart_data[i]
                        continue
                
                else:
                    if "(you)" in chart_data[i]:
                        chart_data[i] = chart_data[i].replace('(you)', '')

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

def post_process(all_leaderboards):
    all_leaderboards = all_leaderboards.dropna(axis=1, how='all')

    all_leaderboards_post = all_leaderboards.copy()
        # Post-processing: Drop the 'Datetime' column and handle merging
    if 'oiwoo' in all_leaderboards_post.columns and 'ooiwoo' in all_leaderboards_post.columns:
        conflict_mask = ~all_leaderboards_post['oiwoo'].isna() & ~all_leaderboards_post['ooiwoo'].isna()
        if conflict_mask.any():
            raise ValueError("Conflict detected: Both 'oiwoo' and 'ooiwoo' have non-NaN values in the same row.")
        all_leaderboards_post['ooiwoo'] = all_leaderboards_post['ooiwoo'].combine_first(all_leaderboards_post['oiwoo'])
        all_leaderboards_post.drop(columns=['oiwoo'], inplace=True)

    # Strip whitespace and periods, then split at commas
    st.write(all_leaderboards_post.index)

    # Remove periods from the index
    all_leaderboards_post.index = all_leaderboards_post.index.str.replace('.', '')

    # Split at the comma and take the second half
    all_leaderboards_post.index = all_leaderboards_post.index.str.split(',', n=1).str[1].str.strip()

    # Split at the space to separate month and day-year
    all_leaderboards_post.index = all_leaderboards_post.index.str.split(' ', n=1)

    # Extract month, day, and year
    month = all_leaderboards_post.index.str[0]
    day_year = all_leaderboards_post.index.str[1].str.split(',')

    print(month, day_year)

    # Combine into a datetime object
    all_leaderboards_post.index = pd.to_datetime(month + ' ' + day_year.str[0] + ', ' + day_year.str[1], format='%b %d, %Y')

    st.write(all_leaderboards_post.index)


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

def fix_mix_ups(mix_ups):

    for key in mix_ups.key():
        for name in mix_ups[key]:
            all_leaderboards_post.drop(columns=[name], inplace=True)

    return True

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
st.markdown("<h1 style='text-align: center; font-weight: bold;'>New York Times Mini - Battle</h1>", unsafe_allow_html=True)
st.markdown("NYT Mini Leaderboard - Tour de France Style")
st.markdown("Best accumulated time over the month wins!")
st.markdown("Upload your screenshots to update the leaderboard")



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

    # Call the function to pivot the DataFrame
    pivoted_results = pivot_leaderboard(results)

    st.markdown("#### Day to Day")
    st.markdown("Note: If the user does not have a result, they are assigned the slowest time")
    st.dataframe(pivoted_results)

    fill_missing = give_missing_worst_time(pivoted_results)

    sum_results = calculate_sum(fill_missing)

    st.markdown("#### Results per day")
    st.dataframe(fill_missing)


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

        # Process the uploaded images
        if uploaded_files:
            st.write("processing...")
            all_leaderboards = extract_leaderboard(uploaded_files)

            all_leaderboards_post = post_process(all_leaderboards)
            st.write("uploading...")

            insert_data(all_leaderboards_post)


finally:

    # Ensure all resources are cleaned up
    grace.close_resources()
    st.session_state.grace = None