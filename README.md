### PROJECT NAVIGATION
cd 4-1p
> The project files are inside the `4-1p` folder. Make sure to navigate into this folder before running any commands.

### VIRTUAL ENVIRONMENT SETUP

### Create Virtual Environment (First time only)
python -m venv venv
> This creates a virtual environment named `venv` to isolate project dependencies.

### Activate Virtual Environment (Every time you work on the project)
venv\Scripts\Activate.ps1
> You'll see `(venv)` appear in your terminal when activation is successful.

##  INSTALL DEPENDENCIES

After activating the virtual environment, install all required packages:
pip install opencv-python opencv-contrib-python numpy pandas pillow


##  VERIFY INSTALLATION

Check if all dependencies are installed correctly:
pip freeze

**Expected output:**
numpy==2.4.3
opencv-contrib-python==4.13.0.92
opencv-python==4.13.0.92
pandas==3.0.1
pillow==12.1.1
python-dateutil==2.9.0.post0
six==1.17.0

## RUN THE PROJECT
python app.py
> This command starts the Attendance Management System.

## ALTERNATIVE METHOD (Without Virtual Environment)

If you don't want to use a virtual environment, you can directly install dependencies:
pip install opencv-python opencv-contrib-python numpy pandas pillow
python app.py

## CONTACT

For any doubts or assistance:

| Name | Email |
|------|-------|
| **U. Vinay** | [vinay11329@gmail.com](mailto:vinay11329@gmail.com) |
| **S. Naga Ganesh** | [somisettynagaganesh@gmail.com](mailto:somisettynagaganesh@gmail.com) |


> **Note:** Always ensure you're inside the `4-1p` folder before running any commands.

