import openai
import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from io import StringIO, BytesIO
import os
import re
from fractions import Fraction
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

openai.api_key = os.getenv("API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all domains, adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # This allows all methods
    allow_headers=["*"],  # This allows all headers
)

# List to store the processed output
processed_outputs = []

def clean_room_name(room):
    room_str = str(room)
    return re.sub(r"#\d+\s*", "", room_str).strip()

async def read_pdf(file) -> str:
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    print("Extracted PDF Text:", text)  # Debug: Check PDF text extraction
    return text

async def read_xlsx(file) -> str:
    content = await file.read()
    xlsx_data = BytesIO(content)
    df = pd.read_excel(xlsx_data)

    # Convert the DataFrame to CSV format for processing, similar to the PDF flow
    csv_data = df.to_csv(index=False)
    print("Extracted XLSX Data:", csv_data)  # Debug: Check XLSX data extraction
    return csv_data

def format_fraction(value):
    try:
        fraction = Fraction(value).limit_denominator(8)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator} {fraction.denominator}/{fraction.denominator}"
    except Exception as e:
        return value

def post_process_dataframe(df):
    required_columns = ['Room', 'Width', 'Height', 'Type', 'Panel', 'Amount', 'Additional', 'Total Labour']

    for column in required_columns:
        if column not in df.columns:
            df[column] = 'N/A' if column in ['Room', 'Type', 'Panel'] else 0
    
    df.fillna({
        'Room': 'N/A',
        'Width': 0,
        'Height': 0,
        'Type': 'N/A',
        'Panel': 'N/A',  # Set panel as empty initially
        'Amount': '',  # Set amount as empty as per the request
        'Additional': '',  # Explicitly set to empty
        'Total Labour': ''  # Explicitly set to empty
    }, inplace=True)
    
    return df

def extract_data_with_gpt4(text, query=None):
    prompt = f"""
    The following text is extracted from a document related to window and door installation. 
    Your task is to extract the following columns from the text and present them in CSV format:
    - Room
    - Width
    - Height
    - Type
    - Panel
    - Additional (leave it empty)
    - Total Labour (leave it empty)
    - Amount

    The "Panel" column should be populated based on the following values:
    - "OX"
    - "XO"
    - "XOX"
    - "SINGLE"
    - "DOUBLE"
    
    For each entry, the panel should be calculated based on the Type:
    - "HORIZONTAL ROLLER XO": OX
    - "HORIZONTAL ROLLER XOX": XOX
    - "SINGLE HUNG": SINGLE
    - "FRENCH DOOR": DOUBLE
    - "DOUBLE FRENCH DOOR": DOUBLE
    - "SLIDING GLASS DOOR XO": XO
    - "SLIDING GLASS DOOR XX": XX
    - If no specific value is found, leave the panel blank.

    For the "Amount", the calculation should be based on the following rules:
    - "HORIZONTAL ROLLER XO": $180
    - "HORIZONTAL ROLLER XOX": $300
    - "SINGLE HUNG": $180
    - "FRENCH DOOR": $450
    - "DOUBLE FRENCH DOOR": $800
    - "SLIDING GLASS DOOR": $250 per panel (X or O count in Type)
    - "FIXED WINDOW": $10 per square foot (Width * Height)
    - "STORE FRONT": $11 per square foot (Width * Height)
    - "SIDE LIGHT": $150

    Make sure the CSV columns align correctly, and leave the "Additional" and "Total Labour" columns empty.
    """
    if query:
        prompt += f"\n\nUser query: {query}"

    prompt += f"\n\nExtracted Text:\n{text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000  
    )

    print("GPT-4 Response:", response)  # Debug: Check GPT-4 response

    if response and 'choices' in response:
        return response['choices'][0]['message']['content'].strip()

    return ""

def parse_csv_data(csv_data):
    try:
        df = pd.read_csv(StringIO(csv_data), on_bad_lines='skip')
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=500, detail="Error parsing CSV data.")

    df = post_process_dataframe(df)
    df["Room"] = df["Room"].apply(clean_room_name)
    df["Width"] = df["Width"].apply(format_fraction)
    df["Height"] = df["Height"].apply(format_fraction)

    # Ensure 'Additional' and 'Total Labour' columns are empty
    df['Additional'] = ''
    df['Total Labour'] = ''

    return df

def append_summary_to_csv(df):
    structured_data = {
        "Column 1": ["Type Of Structure", "Luxury Condo Price", "Construction Type", "Engineering Needed", "Scaffold", 
                     "Caulking and Screws", "Credit Card Fees", "Credit Card Fees Amount", "Shutters", 
                     "Miscellaneous", "Engineering Fees"],
        "Column 2": ["", "", "", "", "", "00", "", "", "", "", ""],
        "Column 3": ["", "Contract Total", "Material Amount", "Material Tax", "Labor Cost", "Total Cost", 
                     "Commission Amount", "Commission Percent", "Profit Percentage", "Profit Amount", "Drive"],
        "Column 4": ["", "$00", "00", "$00", "$00", "$00", "00", "0%", "00%", 
                     "00", ""],
        "Column 5": ["", "HOA", "Permit", "Terms Selection", "Custom Terms", "Custom Terms Notes", "Financing", 
                     "Financing Plan", "", "", ""],
        "Column 6": ["", "y/n", "Y/n", "", "", "", "", "", "", "", ""],
    }
    
    structured_df = pd.DataFrame(structured_data)
    
    # Combine the extracted CSV data with the structured summary
    final_df = pd.concat([df, structured_df], ignore_index=True)
    
    return final_df

@app.post("/uploadfile/")
async def create_upload_file(file: Optional[UploadFile] = File(None), query: Optional[str] = Form(None)):
    global processed_outputs

    if file:
        # File upload path
        if file.filename.endswith('.pdf'):
            text = await read_pdf(file)

            if not text.strip():
                raise HTTPException(status_code=500, detail="No text extracted from PDF.")

            extracted_csv_data = extract_data_with_gpt4(text)
            processed_outputs.append(extracted_csv_data)

        elif file.filename.endswith('.xlsx'):
            csv_data = await read_xlsx(file)
            extracted_csv_data = extract_data_with_gpt4(csv_data)
            processed_outputs.append(extracted_csv_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF or XLSX file.")
        
        df = parse_csv_data(processed_outputs[-1])

        # Append summary to the CSV data (with structured layout)
        df = append_summary_to_csv(df)

        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        return FileResponse(tmp_file_path, filename=f"{os.path.splitext(file.filename)[0]}.csv")

    elif query:
        # Conversation path: handle user query
        if not processed_outputs:
            raise HTTPException(status_code=400, detail="No file has been uploaded yet.")

        latest_output = processed_outputs[-1]
        response = extract_data_with_gpt4(latest_output, query)
        processed_outputs.append(response)

        return JSONResponse(content={"response": response})

    else:
        raise HTTPException(status_code=400, detail="Please upload a file or provide a query.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
