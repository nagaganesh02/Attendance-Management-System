import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directories for storing face samples and the recognizer model
known_faces_dir = 'known_faces'
attendance_file = 'attendance.csv'
ui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UI")

if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

# Function to capture and store face samples
def capture_samples(name, roll_number, sample_count=15):
    cap = cv2.VideoCapture(0)
    count = 0
    valid_samples = 0

    while valid_samples < sample_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print("No faces detected.")
            continue

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]

            face_filename = f"{known_faces_dir}/{name}_{roll_number}_{count+1}.jpg"
            cv2.imwrite(face_filename, face)
            print(f"Captured {face_filename}")

            valid_samples += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Sample {valid_samples}/{sample_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        count += 1
        cv2.imshow('Capturing Samples', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Train the recognizer with the captured samples
def train_recognizer(known_faces_dir):
    face_samples = []
    face_ids = []
    names = {}
    roll_numbers = {}
    current_id = 0

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print(f"No faces detected in {filename}.")
                continue

            for (x, y, w, h) in faces:
                face_samples.append(gray_image[y:y+h, x:x+w])
                face_ids.append(current_id)
                name, roll_number, _ = filename.split('_')
                names[current_id] = name
                roll_numbers[current_id] = roll_number

            current_id += 1

    if len(face_samples) > 0:
        face_recognizer.train(face_samples, np.array(face_ids))
        face_recognizer.save('trained_model.yml')
        print("Training complete.")
    else:
        print("No faces found for training.")

    return names, roll_numbers

# Function to check and mark attendance
def mark_attendance(name, roll_number):
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['Roll Number', 'Name', 'Date', 'Time'])
        df.to_csv(attendance_file, index=False)

    df = pd.read_csv(attendance_file)

    # Ensure roll number column is treated as string
    df['Roll Number'] = df['Roll Number'].astype(str)

    # Check if attendance already recorded for the given roll number and date
    already_marked = df[(df['Roll Number'].str.strip() == roll_number) & (df['Date'] == current_date)]

    if not already_marked.empty:
        return f'Attendance already taken for {name} (Roll No: {roll_number}) today.'

    new_entry = pd.DataFrame({
        'Roll Number': [roll_number],
        'Name': [name],
        'Date': [current_date],
        'Time': [now.strftime('%H:%M:%S')]
    })
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    
    return f'Attendance marked for {name} (Roll No: {roll_number}) at {dt_string}'

# Function to check attendance by student ID with a midnight theme
def check_attendance_by_roll_number():
    # Create a new window for input
    input_window = tk.Toplevel()
    input_window.title("Check Attendance by Roll Number")
    input_window.geometry("400x200")
    input_window.configure(bg="#0d1117")  # Midnight background color

    # Add a label to the input window
    label = tk.Label(input_window, text="Enter Roll Number:", font=("Arial", 14), bg="#0d1117", fg="#c9d1d9")
    label.pack(pady=20)

    # Entry field for roll number
    roll_number_entry = tk.Entry(input_window, font=("Arial", 14))
    roll_number_entry.pack(pady=10)

    # Function to handle submission
    def submit_roll_number():
        roll_number = roll_number_entry.get().strip()  # Get and strip the input

        if not roll_number:
            messagebox.showwarning("Input Error", "Roll number cannot be empty.")
            return

        if not os.path.exists(attendance_file):
            messagebox.showinfo("No Attendance Data", "No attendance data found.")
            return

        df = pd.read_csv(attendance_file)

        # Ensure roll number column is treated as string
        df['Roll Number'] = df['Roll Number'].astype(str)

        # Filter attendance records by roll number
        filtered_df = df[df['Roll Number'].str.strip() == roll_number]

        if filtered_df.empty:
            messagebox.showinfo("No Records", f"No attendance records found for Roll No: {roll_number}.")
            return
        
        # Display records in a new window
        display_attendance_records(filtered_df)

        input_window.destroy()  # Close the input window

    # Submit button
    submit_button = tk.Button(input_window, text="Submit", command=submit_roll_number, font=("Arial", 14), bg="#0288d1", fg="white")
    submit_button.pack(pady=20)

    # Cancel button
    cancel_button = tk.Button(input_window, text="Cancel", command=input_window.destroy, font=("Arial", 14), bg="#d32f2f", fg="white")
    cancel_button.pack(pady=10)

    input_window.focus_set()
    input_window.grab_set()

def display_attendance_records(filtered_df):
    # Create a new window to display the filtered attendance data with a midnight theme
    attendance_window = tk.Toplevel()
    attendance_window.title("Attendance Records")
    attendance_window.geometry("700x400")
    attendance_window.configure(bg="#0d1117")  # Midnight background color

    # Create a treeview to display the data with midnight theme colors
    style = ttk.Style()
    style.theme_use("default")

    # Treeview style with midnight theme colors
    style.configure("Treeview", 
                    background="#0d1117", 
                    foreground="#c9d1d9", 
                    fieldbackground="#0d1117", 
                    font=("Arial", 12))
    
    # Heading style
    style.configure("Treeview.Heading", 
                    background="#161b22", 
                    foreground="#58a6ff", 
                    font=("Arial", 14, "bold"))

    # Scrollbar style
    style.configure("Vertical.TScrollbar", 
                    background="#0d1117", 
                    troughcolor="#161b22", 
                    arrowcolor="#c9d1d9")
    
    style.configure("Horizontal.TScrollbar", 
                    background="#0d1117", 
                    troughcolor="#161b22", 
                    arrowcolor="#c9d1d9")

    tree = ttk.Treeview(attendance_window, columns=list(filtered_df.columns), show='headings', style="Treeview")
    tree.pack(expand=True, fill='both', padx=10, pady=10)

    # Scrollbars without using bg option
    vsb = ttk.Scrollbar(attendance_window, orient="vertical", command=tree.yview, style="Vertical.TScrollbar")
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

    hsb = ttk.Scrollbar(attendance_window, orient="horizontal", command=tree.xview, style="Horizontal.TScrollbar")
    hsb.pack(side='bottom', fill='x')
    tree.configure(xscrollcommand=hsb.set)

    # Define column headings
    for col in filtered_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    # Insert rows into the treeview
    for _, row in filtered_df.iterrows():
        tree.insert("", "end", values=list(row))

    attendance_window.focus_set()

def check_attendance_by_date():
    # Create a new window for input
    input_window = tk.Toplevel()
    input_window.title("Check Attendance by Date")
    input_window.geometry("400x200")
    input_window.configure(bg="#0d1117")  # Midnight background color

    # Add a label to the input window
    label = tk.Label(input_window, text="Enter Date (YYYY-MM-DD):", font=("Arial", 14), bg="#0d1117", fg="#c9d1d9")
    label.pack(pady=20)

    # Entry field for date input
    date_entry = tk.Entry(input_window, font=("Arial", 14))
    date_entry.pack(pady=10)

    # Function to handle submission
    def submit_date():
        date_input = date_entry.get().strip()  # Get and strip the input

        if not date_input:
            messagebox.showwarning("Input Error", "Date cannot be empty.")
            return

        if not os.path.exists(attendance_file):
            messagebox.showinfo("No Attendance Data", "No attendance data found.")
            return

        df = pd.read_csv(attendance_file)

        # Filter attendance records by the provided date
        filtered_df = df[df['Date'] == date_input]

        if filtered_df.empty:
            messagebox.showinfo("No Records", f"No attendance records found for the date: {date_input}.")
            return

        # Sort the filtered records by Roll Number
        sorted_df = filtered_df.sort_values(by='Roll Number')

        # Display sorted records in a new window
        display_attendance_records(sorted_df)

        input_window.destroy()  # Close the input window

    # Submit button
    submit_button = tk.Button(input_window, text="Submit", command=submit_date, font=("Arial", 14), bg="#0288d1", fg="white")
    submit_button.pack(pady=20)

    # Cancel button
    cancel_button = tk.Button(input_window, text="Cancel", command=input_window.destroy, font=("Arial", 14), bg="#d32f2f", fg="white")
    cancel_button.pack(pady=10)

    input_window.focus_set()
    input_window.grab_set()



# Main check attendance function to open the selection window
def check_attendance():
    check_window = tk.Toplevel()
    check_window.title("Check Attendance")
    check_window.geometry("300x100")
    check_window.configure(bg="#040a00")

    tk.Button(check_window, text="By Student ID", command=check_attendance_by_roll_number, bg="#0288d1", fg="white", font=("Arial", 12)).pack(pady=10, padx=20, fill='x')
    tk.Button(check_window, text="By Date", command=check_attendance_by_date, bg="#0288d1", fg="white", font=("Arial", 12)).pack(pady=10, padx=20, fill='x')

# GUI functions
def register_student():
    # Create a pop-up window
    register_window = tk.Toplevel()
    register_window.title("Register Student")
    register_window.geometry("350x500")
    register_window.configure(bg="#090a00")  # Background color matching the theme

    # Add a label at the top (like a title)
    title_label = tk.Label(register_window, text="Register New Student", font=("Arial", 16, "bold"), bg="#090a00", fg="#f8e11b")
    title_label.pack(pady=20)

    # Create labels and entry fields for Name and Roll Number
    name_label = tk.Label(register_window, text="Name:", font=("Arial", 14), bg="#090a00", fg="#f8e11b")
    name_label.pack(pady=10)
    name_entry = tk.Entry(register_window, font=("Arial", 14))
    name_entry.pack(pady=5)

    roll_label = tk.Label(register_window, text="Roll Number:", font=("Arial", 14), bg="#090a00", fg="#f8e11b")
    roll_label.pack(pady=10)
    roll_entry = tk.Entry(register_window, font=("Arial", 14))
    roll_entry.pack(pady=5)

    # Button to capture samples and register
    def submit_registration():
        name = name_entry.get()
        roll_number = roll_entry.get()
        if name and roll_number:
            capture_samples(name, roll_number)
            train_recognizer(known_faces_dir)

            # Create a new window for displaying the image and message
            success_window = tk.Toplevel()
            success_window.title("Registration Complete")
            success_window.geometry("400x500")
            success_window.configure(bg="#0d1117")  # Midnight background color

            # Load and display the image
            img_path = os.path.join(ui_dir, "0001.png")
            img = Image.open(img_path)
            img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Resize the image as needed
            photo = ImageTk.PhotoImage(img)

            img_label = tk.Label(success_window, image=photo, bg="#0d1117")
            img_label.pack(pady=20)  # Added padding for better layout

            # Keep reference to the image to prevent garbage collection
            img_label.image = photo

            # Display the success message with custom styling
            message = "Registration complete!\nYou can now take attendance."
            message_label = tk.Label(success_window, text=message, font=("Arial", 16, "bold"), bg="#0d1117", fg="#58a6ff")
            message_label.pack(pady=20)

            # Add a button to close the window
            close_button = tk.Button(success_window, text="Close", font=("Arial", 14), bg="#28a745", fg="white", command=success_window.destroy)
            close_button.pack(pady=20)

            success_window.focus_set()
            success_window.grab_set()

            register_window.destroy()  # Close the registration window after showing the success window
        else:
            messagebox.showwarning("Input Error", "Name and roll number cannot be empty.")

    # Register button with improved styling
    register_button = tk.Button(register_window, text="Submit", font=("Arial", 14), bg="#0288d1", fg="white", command=submit_registration)
    register_button.pack(pady=30)

    # Cancel button with improved styling
    cancel_button = tk.Button(register_window, text="Cancel", font=("Arial", 14), bg="#d32f2f", fg="white", command=register_window.destroy)
    cancel_button.pack(pady=10)

    register_window.focus_set()
    register_window.grab_set()



def take_attendance():
    if not os.path.exists('trained_model.yml'):
        messagebox.showerror("Error", "No trained model found. Please register students first.")
        return

    names, roll_numbers = train_recognizer(known_faces_dir)
    face_recognizer.read('trained_model.yml')

    video_capture = cv2.VideoCapture(0)
    marked_names = set()
    message = ""

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_id, confidence = face_recognizer.predict(gray_frame[y:y+h, x:x+w])
            name = names.get(face_id, "Unknown")
            roll_number = roll_numbers.get(face_id, "Unknown")

            if name != "Unknown":
                message = mark_attendance(name, roll_number)

                if (name, roll_number) not in marked_names:
                    marked_names.add((name, roll_number))
                    cv2.putText(frame, message, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    break

        if len(marked_names) > 0:
            break

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if message:
        # Create a new window for displaying the image and message
        success_window = tk.Toplevel()
        success_window.title("Attendance Taken")
        success_window.geometry("300x350")
        success_window.configure(bg="#040a00")

        img_path = os.path.join(ui_dir, "0001.png")
        img = Image.open(img_path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)  # Resize the image as needed
        photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(success_window, image=photo, bg="#040a00")
        img_label.pack(pady=10)

        # Keep reference to the image to prevent garbage collection
        img_label.image = photo       

        # Create a canvas to hold the scrolling message
        canvas = tk.Canvas(success_window, bg="#040a00", width=300, height=100)
        canvas.pack()

        # Create a text item on the canvas with the message
        text_id = canvas.create_text(75, 75, text=message, font=("Arial", 14), fill="#f4dc0c", anchor="w")

        # Function to animate the scrolling text
        def scroll_text():
            x_pos = canvas.bbox(text_id)[2]  # Right edge of the text
            if x_pos > -canvas.winfo_width():
                canvas.move(text_id, -3, 0)
                success_window.after(10, scroll_text)
            else:
                # Reset position for continuous scrolling
                canvas.coords(text_id, canvas.winfo_width(), 75)
                scroll_text()

        # Start scrolling
        scroll_text()

        success_window.mainloop()  # Keep the window open until closed by the user

# Create the main application window
def main_window():
    root = tk.Tk()
    root.title("Attendance System")
    root.geometry("400x700")  # Adjust height to accommodate three images and buttons
    root.configure(bg="#040a00")

    # Create a central frame for better alignment
    central_frame = tk.Frame(root, bg="#040a00")
    central_frame.pack(expand=True)

    # Configure grid in the central frame to center the widgets
    central_frame.grid_rowconfigure(0, weight=1)
    central_frame.grid_rowconfigure(1, weight=1)
    central_frame.grid_rowconfigure(2, weight=1)
    central_frame.grid_rowconfigure(3, weight=1)
    central_frame.grid_rowconfigure(4, weight=1)
    central_frame.grid_columnconfigure(0, weight=1)

    # Load and display the first image and button
    img_path1 = os.path.join(ui_dir, "register.png")
    img1 = Image.open(img_path1)
    img1 = img1.resize((150, 150), Image.Resampling.LANCZOS)  # Resize the image as needed
    main_image1 = ImageTk.PhotoImage(img1)
    
    img_label1 = tk.Label(central_frame, image=main_image1, bg="#f7f7f7")
    img_label1.grid(row=0, column=0, pady=10)

    # Create a frame for the "Register Student" button with a border
    register_border_frame = tk.Frame(central_frame, highlightbackground="#8c908c", highlightthickness=2)
    register_border_frame.grid(row=1, column=0, pady=10)
    
    register_button = tk.Button(register_border_frame, text="Register Student", command=register_student, font=("Arial", 14), bg="#0c0e0b", fg="yellow", activebackground="#4d4d4d", relief="flat")
    register_button.pack(padx=2, pady=2)

    # Load and display the second image and button
    img_path2 = os.path.join(ui_dir, "attendance.png")
    img2 = Image.open(img_path2)
    img2 = img2.resize((150, 150), Image.Resampling.LANCZOS)  # Resize the image as needed
    main_image2 = ImageTk.PhotoImage(img2)
    
    img_label2 = tk.Label(central_frame, image=main_image2, bg="#f7f7f7")
    img_label2.grid(row=2, column=0, pady=10)

    # Create a frame for the "Take Attendance" button with a border
    attendance_border_frame = tk.Frame(central_frame, highlightbackground="#8c908c", highlightthickness=2)
    attendance_border_frame.grid(row=3, column=0, pady=10)
    
    take_attendance_button = tk.Button(attendance_border_frame, text="Take Attendance", command=take_attendance, font=("Arial", 14), bg="#0c0e0b", fg="yellow", activebackground="#4d4d4d", relief="flat")
    take_attendance_button.pack(padx=2, pady=2)

    # Load and display the third image and button
    img_path3 = os.path.join(ui_dir, "verifyy.png")
    img3 = Image.open(img_path3)
    img3 = img3.resize((150, 150), Image.Resampling.LANCZOS)  # Resize the image as needed
    main_image3 = ImageTk.PhotoImage(img3)
    
    img_label3 = tk.Label(central_frame, image=main_image3, bg="#f7f7f7")
    img_label3.grid(row=4, column=0, pady=10)

    # Create a frame for the "Check Attendance" button with a border
    check_border_frame = tk.Frame(central_frame, highlightbackground="#8c908c", highlightthickness=2)
    check_border_frame.grid(row=5, column=0, pady=10)
    
    check_attendance_button = tk.Button(check_border_frame, text="Check Attendance", command=check_attendance, font=("Arial", 14), bg="#0c0e0b", fg="yellow", activebackground="#4d4d4d", relief="flat")
    check_attendance_button.pack(padx=2, pady=2)

    # Keep references to the images to prevent garbage collection
    img_label1.image = main_image1
    img_label2.image = main_image2
    img_label3.image = main_image3

    root.mainloop()

if __name__ == "__main__":
    main_window()
