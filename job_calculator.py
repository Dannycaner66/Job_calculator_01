import openai
import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import StringIO, BytesIO
import os
import re
from fractions import Fraction
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

openai.api_key = os.getenv("API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return text

async def read_xlsx(file) -> str:
    content = await file.read()
    xlsx_data = BytesIO(content)
    df = pd.read_excel(xlsx_data)

    csv_data = df.to_csv(index=False)
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
        'Panel': 'N/A',
        'Amount': '',
        'Additional': '',
        'Total Labour': ''
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

    Make sure the columns align correctly.
    """
    if query:
        prompt += f"\n\nUser query: {query}"

    prompt += f"\n\nExtracted Text:\n{text}"

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

    df['Additional'] = ''
    df['Total Labour'] = ''

    return df

def convert_df_to_json(df):
    # Convert the DataFrame to a list of dictionaries
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "Room": row["Room"],
            "Width": row["Width"],
            "Height": row["Height"],
            "Type": row["Type"],
            "Panel": row["Panel"],
            "Amount": row["Amount"],
            "Additional": row["Additional"],
            "Total Labour": row["Total Labour"]
        })
    return data_list

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

        # Convert DataFrame to JSON format
        json_data = convert_df_to_json(df)

        return JSONResponse(content=json_data)

    elif query:
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
