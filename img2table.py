import pandas as pd
import cv2
import easyocr
import os
import numpy as np
import streamlit as st
import tempfile
from sshtunnel import SSHTunnelForwarder
import time
import psycopg2

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
        self.close_resources(self)
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
        self.close_resources(self)
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


# Track user activity
if 'last_active' not in st.session_state:
    st.session_state.last_active = time.time()

# Update activity timestamp
st.session_state.last_active = time.time()

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

    test = getQuery("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'user_data';
            """)
    
    st.write(test)

finally:

    # Ensure all resources are cleaned up
    grace.close_resources()
    st.session_state.grace = None