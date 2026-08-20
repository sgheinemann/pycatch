SBCross
=======

Welcome to **SBCross**: The Python Framework for Sector Boundary Analysis

SBCross is a python-based processing engine and analysis pipeline built to extract, characterize, and analyze Sector Boundary (SB) crossings in the solar wind. 

This repository contains the backend code and algorithmic framework used to generate, process, and maintain the **SBCross Database**, which is hosted and publicly accessible at:
👉 **[N/A]** 

Using in-situ plasma, magnetic field, and suprathermal electron observations (such as OMNI and Wind data), SBCross provides a standardized toolkit to process magnetic field polarities, analyze electron pitch angle distributions (PAD), evaluate crossing durations, and calculate physical parameters such as Parker spiral-adjusted sector boundary thickness.

The framework supports multi-mission observational data covering historical to modern solar cycles (1995–present), offering both high-resolution parameter extraction and interactive, reproducible event analysis.


Features
--------
* **Sector Boundary Identification:** Standardized pipeline to detect true sector boundary polarity reversals, crossing health, and Svalgaard in-situ polarity markers.
* **Plasma & Field Dynamics:** Automated extraction and statistical profiling (means, standard deviations, peaks) of solar wind bulk speed ($V$), proton density ($N$), magnetic field magnitude ($B$), and plasma beta ($\beta$).
* **Parker Geometry Integration:** Dynamical calculation of local Parker spiral angles and Sector Boundary physical thickness ($L = V_{\mathrm{normal}} \cdot \Delta t$).
* **Visualization Engine:** Automated generation of high-quality 5-panel event diagnostic plots including magnetic components, electron asymmetry indices, and pitch angle spectrograms.


Setup
--------------------
Download the Jupyter Notebook, install dependencies and run.

