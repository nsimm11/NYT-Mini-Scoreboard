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


def extract_leaderboard(uploaded_files):

    all_leaderboards = pd.DataFrame()  # Initialize an empty DataFrame
    for uploaded_file in uploaded_files:
        st.write("processing image...")
        
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
                continue  # Skip to the next file if decoding fails

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

        # Construct the path to the user's profile picture
        user_photo_path = f"user_photos/{username.strip()}/profile_picture.jpg"  # Adjust the extension as needed

        # Check if the profile image exists
        if os.path.exists(user_photo_path):
            # Use the profile image
            profile_html = f"<img src='{user_photo_path}' style='width: 50px; height: 50px; border-radius: 50%;'>"
        else:
            # Generate a random color for each user
            color = "#{:06x}".format(np.random.randint(0, 0xFFFFFF))
            # Use a colored circle with the first letter of the username
            first_letter = username[0].upper()
            profile_html = f"<div style='width: 50px; height: 50px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>{first_letter}</div>"

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
            file_size = uploaded_file.size / (1024 * 1024)
            if file_size > 0.2:
                st.write(f"File size of {uploaded_file.name} is {file_size:.2f} MB, too large...")
                # If the file size is above 0.2 MB, resize the image
                image = Image.open(uploaded_file)
                # Define the new size (you can adjust this as needed)
                new_size = (800, 800)  # Resize to 800x800 pixels
                image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS for high-quality downsampling

                # Save the resized image to a bytes buffer
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')  # Save as PNG or any other format
                img_byte_arr.seek(0)  # Move to the beginning of the BytesIO buffer

                # Calculate the size of the resized image
                reduced_size = img_byte_arr.tell() / (1024 * 1024)  # Size in MB
                st.write(f"Reduced size of {uploaded_file.name} is {reduced_size:.2f} MB.")

                # Use the resized image for further processing
                uploaded_file = img_byte_arr  # Update the uploaded_file to the resized image

            # Process the image (either original or resized)
            # Example: all_leaderboards = extract_leaderboard([uploaded_file])

        # Process the uploaded images
        if uploaded_files:
            if len(st.session_state["final_data"]) == 0:
                try:
                    all_leaderboards = extract_leaderboard(uploaded_files)

                    all_leaderboards_post = post_process(all_leaderboards)

                    st.session_state["final_data"] = all_leaderboards_post
                except Exception as e:
                    all_leaderboards_post = pd.DataFrame()
                    st.warning(f"Failed to Process Image: {str(e)}")

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
                        
                        st.write("uploading...")

                        insert_data(all_leaderboards_post)

                        st.session_state["final_data"] = pd.DataFrame()

                    except Exception as e:
                        st.warning(f"Failed to Push to DB: {str(e)}")
                        st.session_state["final_data"] = pd.DataFrame()


    
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



finally:
    # Ensure all resources are cleaned up
    grace.close_resources()
    st.session_state.grace = None