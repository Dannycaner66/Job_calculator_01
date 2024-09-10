import openai
import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from io import StringIO
import os
import re
from fractions import Fraction
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("API_KEY")

app = FastAPI()

def clean_room_name(room):
    room_str = str(room)
    return re.sub(r"#\d+\s*", "", room_str).strip()

async def read_pdf(file) -> str:
    content = await file.read()
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def format_fraction(value):
    try:
        fraction = Fraction(value).limit_denominator(8)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator} {fraction.denominator}/{fraction.denominator}"
    except Exception as e:
        return value

def post_process_dataframe(df):
    required_columns = ['Room', 'Width', 'Height', 'Type', 'Panel', 'Price', 'Additional Labour', 'Total Labour']

    for column in required_columns:
        if column not in df.columns:
            df[column] = 'N/A' if column in ['Room', 'Type', 'Panel', 'Additional Labour'] else 0
    
    df.fillna({
        'Room': 'N/A',
        'Width': 0,
        'Height': 0,
        'Type': 'N/A',
        'Panel': 'N/A',
        'Price': 0,
        'Additional Labour': 'N/A',
        'Total Labour': 0
    }, inplace=True)
    
    return df

def extract_data_with_gpt4(pdf_text):
    prompt = f"""
    The following text is extracted from a PDF document related to window and door installation. 
    Your task is to extract the following columns from the text and present them in CSV format:
    - Room
    - Width
    - Height
    - Type
    - Panel
    - Additional Labour
    - Total Labour Cost
    - Price

    For each entry, the price should be calculated based on the following rules:
    - "HORIZONTAL ROLLER XO": $180
    - "HORIZONTAL ROLLER XOX": $300
    - "SINGLE HUNG": $180
    - "FRENCH DOOR": $450
    - "DOUBLE FRENCH DOOR": $800
    - "SLIDING GLASS DOOR": $250 per panel (X or O count in Type)
    - "FIXED WINDOW": $10 per square foot (Width * Height)
    - "STORE FRONT": $11 per square foot (Width * Height)
    - "SIDE LIGHT": $150

    If any column does not have a value (such as Panel or Additional Labour), write "N/A" or leave it blank accordingly.
    
    Make sure the CSV columns align correctly. The CSV should contain:
    - Room
    - Width
    - Height
    - Type
    - Panel
    - Price
    - Additional Labour
    - Total Labour
    
    Extracted PDF Text:
    {pdf_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2000  
    )

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

    return df

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")

    pdf_text = await read_pdf(file)
    extracted_csv_data = extract_data_with_gpt4(pdf_text)

    if extracted_csv_data:
        df = parse_csv_data(extracted_csv_data)

        if df.empty:
            raise HTTPException(status_code=500, detail="Extracted data is empty or invalid.")

        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

        return FileResponse(tmp_file_path, filename=f"{os.path.splitext(file.filename)[0]}.csv")
    else:
        raise HTTPException(status_code=500, detail="Failed to extract data from the PDF.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
