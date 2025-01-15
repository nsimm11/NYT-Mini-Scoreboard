import streamlit as st
import pandas as pd
import easyocr
import numpy as np
import cv2
import tempfile
from sshtunnel import SSHTunnelForwarder
import time
import psycopg2
import os

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


@st.cache_data
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

            print(text_lines[4:3+6*2-1])

            datetime_str = text_lines[3]

            # Create a dictionary to hold the leaderboard data
            leaderboard_dict = {"Datetime": datetime_str}

            # Populate the dictionary with usernames as keys and their values
            for i in range(4, 2+6*2+1, 2):
                if "(you)" in text_lines[i]:
                    text_lines[i] = text_lines[i].replace(" (you)", "")
                time_str = text_lines[i+1]
                # Calculate total seconds from time_str
                if "." in time_str:
                    minutes, seconds = map(int, time_str.split('.'))
                    total_seconds = minutes * 60 + seconds
                elif ":" in time_str:
                    minutes, seconds = map(int, time_str.split(':'))
                    total_seconds = minutes * 60 + seconds
                else:
                    print("Error: Time format not recognized", time_str)
                    total_seconds = np.nan
                
                leaderboard_dict[text_lines[i]] = total_seconds

            # Convert to a DataFrame and transpose it
            leaderboard_df = pd.DataFrame(leaderboard_dict, index=[datetime_str])

            # Append the current leaderboard to the all_leaderboards DataFrame
            all_leaderboards = pd.concat([all_leaderboards, leaderboard_df])

    return all_leaderboards

def post_process(all_leaderboards):
    all_leaderboards_post = all_leaderboards.copy()
        # Post-processing: Drop the 'Datetime' column and handle merging
    all_leaderboards_post.drop(columns=["Datetime"], inplace=True)
    if 'oiwoo' in all_leaderboards_post.columns and 'ooiwoo' in all_leaderboards_post.columns:
        conflict_mask = ~all_leaderboards_post['oiwoo'].isna() & ~all_leaderboards_post['ooiwoo'].isna()
        if conflict_mask.any():
            raise ValueError("Conflict detected: Both 'oiwoo' and 'ooiwoo' have non-NaN values in the same row.")
        all_leaderboards_post['ooiwoo'] = all_leaderboards_post['ooiwoo'].combine_first(all_leaderboards_post['oiwoo'])
        all_leaderboards_post.drop(columns=['oiwoo'], inplace=True)

    # Strip whitespace and convert the index to datetime using the specific format
    all_leaderboards_post.index = pd.to_datetime(all_leaderboards_post.index.str.strip(), format='mixed')


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
        db_user=st.secrets["postgres"]["username_post"],
        db_password=st.secrets["postgres"]["password_post"],
        db_name=st.secrets["postgres"]["database_post"],
        db_host=st.secrets["postgres"]["hostname"],
        db_port=st.secrets["postgres"]["port"]
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
    
    st.dataframe(results)

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