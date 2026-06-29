---
Title: Rotor Topology and Motor Size Selection with Geometry Optimization of Synchronous Reluctance Motors (SynRM) Using Hybrid Machine Learning
Subtitle: Setup and Operating Guide
Author: Soham Wankhade
Version: 1.0
Document_type: Technical README
---

# Rotor Topology and Motor Size Selection with Geometry Optimization of
Synchronous Reluctance Motors (SynRM) Using Hybrid Machine Learning

## Setup and Operating Guide

> **Document Information**  
> **Author:** Soham Wankhade  
> **Version:** 1.0  
> **Document Type:** Technical README  
> **Project:** Rotor Topology and Motor Size Selection with Geometry Optimization of Synchronous Reluctance Motors (SynRM) Using Hybrid Machine Learning  
> **Status:** Draft

========================================================================
Rotor Topology and Motor Size Selection with Geometry Optimization of
Synchronous Reluctance Motors (SynRM) Using Hybrid Machine Learning
------------------------------------------------------------------------
Setup and Operating Guide
========================================================================

Welcome to the SynRM Rotor Topology and Geometry Optimization Pipeline.
This tool combines hybrid machine learning (XGBoost and CatBoost) with
the NSGA-II evolutionary algorithm to identify optimal rotor topologies
and motor sizing for Synchronous Reluctance Motors, based on your own
Altair HyperStudy simulation data.

This guide walks you through setting up and running the software on a
new computer, from a clean install to your first optimized design.

*** IMPORTANT SYSTEM REQUIREMENT ***
Python Version: Python 3.12 or newer is STRICTLY REQUIRED.
************************************

------------------------------------------------------------------------
PHASE 1: Prerequisites (Software Installation)
------------------------------------------------------------------------
1. Install Python
   - Go to python.org/downloads and download Python 3.12 (or newer).
   - Run the installer.
   - CRITICAL STEP: At the bottom of the first installation screen, you
     MUST check the box that says "Add Python.exe to PATH" before
     clicking Install.

2. Install Visual Studio Code (VS Code)
   - Go to code.visualstudio.com and download the installer.
   - Install it using the default settings.

------------------------------------------------------------------------
PHASE 2: Folder Management & Directory Structure
------------------------------------------------------------------------
For the pipeline to work, your files must be organized exactly like this.
Your main code lives inside the "Motor_optimizer" folder, alongside your
raw Altair HyperStudy output folders.

Motor_optimizer/                  <-- YOUR MAIN PROJECT FOLDER
├── data/                         <-- Copy the .data files here WITH MAPPING
├── models/                       <-- AI models are saved here automatically
├── results/                      <-- Global metrics and scorecards saved here
│   ├── design/                   <-- Final optimized geometry data saved here
│   └── plots/                    <-- Visual scatter plots and charts saved here
├── utils/
│   ├── __pycache__/              <-- Auto-generated Python cache (safely ignore)
│   └── data_parser.py            <-- Helper script that reads HyperStudy files
├── optimize.py                   <-- The NSGA-II optimizer script
├── train.py                      <-- The AI training script
├── requirements.txt              <-- Python dependencies list
└── README.MD                     <-- This instruction guide

------------------------------------------------------------------------
PHASE 3: Project Setup
------------------------------------------------------------------------
1. Move the Project: Ensure your entire "Motor_optimizer" folder
   (as shown above) is copied onto your new PC.
2. Open in VS Code:
   - Open Visual Studio Code.
   - Click on File > Open Folder... in the top left menu.
   - Locate and select the "Motor_optimizer" folder. Click "Select Folder".
   - (If asked to trust the authors, check the box and click "Yes").

------------------------------------------------------------------------
PHASE 4: The Virtual Environment (Clean Slate)
------------------------------------------------------------------------
We use a Virtual Environment to install the project's dependencies
safely without affecting your computer's main Python installation.

1. Open the Terminal inside VS Code by clicking Terminal > New Terminal
   at the top menu.
2. Create the virtual environment by typing this command and hitting Enter:
   python -m venv venv

3. Activate the environment:
   Type the command for your specific operating system and hit Enter:

   For Windows:
   .\venv\Scripts\activate

   For Mac / Linux:
   source venv/bin/activate

SUCCESS CHECK: You should now see "(venv)" appear at the very beginning
of your terminal command prompt.

(Windows Troubleshooting: If you get a red error saying "running scripts
is disabled", type this command: Set-ExecutionPolicy Unrestricted -Scope Process
Hit Enter, and then try the activate command again).

------------------------------------------------------------------------
PHASE 5: Install Dependencies
------------------------------------------------------------------------
1. Ensure "(venv)" is showing in your terminal.
2. Run the following command:
   pip install -r requirements.txt
3. Wait for the installation to finish.

------------------------------------------------------------------------
PHASE 6: Data Preparation (Crucial Step)
------------------------------------------------------------------------
The AI needs your Altair HyperStudy data to learn.

1. In VS Code, open the "data" folder inside "Motor_optimizer".
2. Go to your raw HyperStudy folders (e.g., ivr_RFB_C3_A0(11t15)), find
   the ".data" runsummary files, and copy them into the "Motor_optimizer/data"
   folder.
3. THE MAPPING RULE: You must open every single ".data" file in VS Code
   and ensure the custom mapping instructions are pasted at the very top.

   Example Mapping Header:
   # MAP | ivr_RFB_C3_A11 | var_1, var_2, var_3, var_4, var_5, var_6, var_7 | r_1, r_2, r_3, r_4
   # MAP | ivr_RFB_C3_A12 | var_8, var_9, var_10, var_11, var_12, var_13, var_14 | r_5, r_6, r_7, r_8

*** The Power of Flexible Mapping ***
   This mapping system is one of the best features of the code. If any geometry
   within a batch file is unsuitable for training (due to failed simulations
   or bad data), you can simply delete its "# MAP" line from the top of the file.
   The software will completely ignore it.

   If you re-run that specific failed geometry separately later, just drop the
   new, separate .data file into the "data" folder and map that single geometry
   at the top of the new file. The AI will automatically stitch everything
   together.

------------------------------------------------------------------------
PHASE 7: Running the Software
------------------------------------------------------------------------
Make sure your terminal still shows "(venv)" before running these.

Step 1: Train the AI Models
Run this command to analyze your data and build the XGBoost and CatBoost models:
   python train.py

Step 2: Run the Optimizer
Run this command to execute the NSGA-II evolutionary algorithm. This will
use the trained models to find the Pareto-optimal rotor geometries matching
your specific performance targets and tolerances:

   python optimize.py --torque 55.0 --efficiency 90.5 --power_factor 0.6 --ripple 20.0 --torque_tol 4.0 --efficiency_tol 5 --power_factor_tol 0.2 --ripple_tol 5 --top_n 8

(Note: The "--top_n 8" parameter at the very end specifies the total number of
top optimized geometries you want the software to output. You can easily change
this number to 5, 10, or however many final designs you need).

------------------------------------------------------------------------
PHASE 8: Manufacturable Airgap Range (Optional)
------------------------------------------------------------------------
When train.py analyzes your data, it automatically detects whether the
airgap in your training data is FIXED (only one gap was ever tested) or
VARIABLE (your data spans a real range of gaps). This is reported during
training, for example:

   ↳ D03: airgap detected as VARIABLE, 0.350-0.900mm radial (from data)

If your geometry's airgap is VARIABLE, you can instruct the optimizer to
search only within the gap range your manufacturing process can actually
hold, using the --airgap_min and --airgap_max flags (both are DIAMETRIC
values in mm, and must be provided together):

   python optimize.py --torque 55.0 --efficiency 90.5 --power_factor 0.6 --ripple 20.0 --airgap_min 1.0 --airgap_max 1.8 --top_n 8

What happens with these flags:
- The optimizer will only search within your requested range, intersected
  with whatever range the AI actually observed during training (it will
  never select a gap outside that range, since predictions there would be
  unreliable extrapolations).
- If your requested range is wider than what the data supports, it is
  automatically clipped, and a message is printed indicating the usable
  range that was applied.
- If your geometry's airgap was detected as FIXED (not variable), these
  flags are ignored and a warning is printed — a fixed-gap geometry only
  ever had one airgap value in its training data, so there is no range to
  select from until the model is retrained on varied-gap data.
- If only one of --airgap_min / --airgap_max is provided (not both), the
  optimizer will stop and prompt you to provide both.

========================================================================
