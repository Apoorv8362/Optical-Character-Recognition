Objective
Perform OCR on two provided images containing historical prices for:
	•	BEHLTREU (bond index) and
	•	AHWM5 (future on BEHLTREU)
to extract:
	•	date
	•	bond index price
	•	future price
and compile them into a clean, structured CSV for further analysis.

Approach - 
1. OCR Extraction
	•	Utilised Tesseract OCR (via pytesseract) with the Pillow package in Python for image pre-processing and text extraction.
	•	Applied resizing and grayscale filtering to improve OCR accuracy.
	•	Manually reviewed and corrected any dates or values not correctly captured by OCR to ensure data integrity.
2. Data Cleaning - 
	•	Parsed extracted text into structured columns: date, bond_index_price, future_price.
	•	Converted prices to numeric types and dates to a consistent YYYY-MM-DD format.
	•	Removed duplicates and handled missing values appropriately. 
	•	10 data fields have been manually added after OCR implementation.
3. CSV Generation
	•	Exported the cleaned dataset as:  behltreu_ahwm5_data.csv   for use in Question 2 (Stress Scenario Modeling).

Files Included
	•	ocr_script.py: Python script performing OCR, data cleaning, and CSV export.
	•	behltreu_ahwm5_cleaned.csv: Final cleaned CSV with aligned dates, bond index prices, and future prices.
	•	sample_ocr_output_preview.png: (Optional) Image showing a preview of the cleaned output for reference.

Dependencies
	•	pytesseract
	•	Pillow
	•	pandas
Install dependencies using:

!pip install pytesseract pillow pandas numpy
!pip install pillow

How to Run
	1	Ensure Tesseract is installed on your system and added to your PATH:
	◦	Tesseract Installation Guide
	2	Place the image files:
	◦	behltreu.png
	◦	ahwm5.png in the same directory as ocr_script.py.
	3	Run:
		python ocr_script.py
	4	The script will generate behltreu_ahwm5_cleaned.csv ready for downstream analysis.

Key Notes
 OCR was validated manually for accuracy where parsing errors occurred.
Data is aligned and cleaned to support stress scenario modeling in Question 2.
The workflow can be adapted for batch processing additional index images if required in production.
