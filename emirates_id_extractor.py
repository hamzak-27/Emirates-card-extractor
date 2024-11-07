import boto3
import json
from typing import Dict, Optional
import os
from IPython.display import display, HTML
import re
from datetime import datetime
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

class EmiratesIDExtractor:
    def __init__(self, region_name: str, aws_access_key_id: str, aws_secret_access_key: str, openai_api_key: str):
        """Initialize the Emirates ID Extractor with AWS and OpenAI credentials."""
        self.textract_client = boto3.client(
            "textract",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        self.s3_client = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize LangChain components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)

    def upload_to_s3(self, local_file_path: str, bucket_name: str, s3_file_path: str) -> str:
        """Upload the image to S3 bucket."""
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
            return f"s3://{bucket_name}/{s3_file_path}"
        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")

    def process_and_query(self, text: str) -> Dict[str, str]:
        """Process extracted text using LangChain and OpenAI."""
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            
            # Create vector store
            docsearch = FAISS.from_texts(texts, self.embeddings)
            
            # Create QA chain
            chain = load_qa_chain(self.llm, chain_type="stuff")
            
            # Modified prompt with better guidance for profession and sponsor
            query = """Extract the following details from the text. For each field, if the information is not found, write 'Not Found'.
            
            1. Name (full name in English)
            2. UID No. (Emirates ID number)
            3. Passport No.
            4. Profession:
               - If you see "Partner" or "Partner (Female)" by itself, this is the Profession
               - Simple job titles should be listed as Profession
            5. Sponsor:
               - If you see company names like "MACKSOFY DATA MANAGEMENT & CYBER SECURITY SERVICES CO.", this is the Sponsor
               - Company names should always be listed as Sponsor, not Profession
            6. Place of Issue (should be a city name like Dubai)
            7. Issue Date (in format YYYY/MM/DD)
            8. Expiry Date (in format YYYY/MM/DD)

            Important:
            - Any text containing 'SERVICES CO', 'DATA MANAGEMENT' should be listed as Sponsor
            - Job titles like 'Partner' or 'Partner (Female)' should be listed as Profession
            
            Format the response as a JSON object with these exact keys:
            {
                "name": "",
                "uid_no": "",
                "passport_no": "",
                "profession": "",
                "sponsor": "",
                "place_of_issue": "",
                "issue_date": "",
                "expiry_date": ""
            }
            
            Only return the JSON object, nothing else."""
            
            # Get relevant documents
            docs = docsearch.similarity_search(query)
            
            # Run the query
            result = chain.run(input_documents=docs, question=query)
            
            # Parse the result into a dictionary
            try:
                extracted_info = json.loads(result)
                
                # Add post-processing for profession/sponsor
                if 'profession' in extracted_info and 'sponsor' in extracted_info:
                    prof_value = extracted_info['profession'].upper()
                    sponsor_value = extracted_info['sponsor'].upper()
                    
                    # Check if fields need to be swapped
                    if 'SERVICES CO' in prof_value or 'DATA MANAGEMENT' in prof_value:
                        # Swap the fields
                        extracted_info['profession'], extracted_info['sponsor'] = sponsor_value, prof_value
                    
                    # If sponsor contains just partner, swap it
                    if sponsor_value.strip() in ['PARTNER', 'PARTNER (FEMALE)']:
                        extracted_info['profession'], extracted_info['sponsor'] = sponsor_value, prof_value
                        
            except json.JSONDecodeError:
                extracted_info = self.extract_using_regex(result)
            
            return extracted_info
            
        except Exception as e:
            raise Exception(f"Error in LLM processing: {str(e)}")

    def extract_using_regex(self, text: str) -> Dict[str, str]:
        """Fallback method to extract information using regex patterns."""
        extracted_info = {
            "name": "Not Found",
            "uid_no": "Not Found",
            "passport_no": "Not Found",
            "profession": "Not Found",
            "sponsor": "Not Found",
            "place_of_issue": "Not Found",
            "issue_date": "Not Found",
            "expiry_date": "Not Found"
        }
        
        # Extract information using regex patterns
        # Place of Issue pattern
        place_match = re.search(r'(Dubai|Abu Dhabi|Sharjah|Ajman|Umm Al Quwain|Ras Al Khaimah|Fujairah)',
                              text, re.IGNORECASE)
        if place_match:
            extracted_info["place_of_issue"] = place_match.group(1)
        
        # UID pattern (numbers only)
        uid_match = re.search(r'\b\d{8,9}\b', text)
        if uid_match:
            extracted_info["uid_no"] = uid_match.group(0)
        
        # Passport pattern (Z followed by numbers)
        passport_match = re.search(r'Z\d{7}', text)
        if passport_match:
            extracted_info["passport_no"] = passport_match.group(0)
        
        # Date pattern
        date_matches = re.findall(r'\d{4}/\d{2}/\d{2}', text)
        if len(date_matches) >= 2:
            extracted_info["issue_date"] = date_matches[0]
            extracted_info["expiry_date"] = date_matches[1]
        
        # Name pattern (all caps words)
        name_match = re.search(r'([A-Z]+\s+){2,}[A-Z]+', text)
        if name_match:
            extracted_info["name"] = name_match.group(0)
            
        # Professional pattern
        if 'PARTNER' in text or 'Partner' in text:
            extracted_info["profession"] = "Partner"
            
        # Sponsor pattern (company name)
        sponsor_match = re.search(r'MACKSOFY.*SERVICES CO\.', text)
        if sponsor_match:
            extracted_info["sponsor"] = sponsor_match.group(0)
        
        return extracted_info

    def extract_text_from_image(self, image_path: str, bucket_name: str) -> Dict[str, str]:
        """Extract text from Emirates ID image using Amazon Textract and process with LLM."""
        try:
            # Upload image to S3
            s3_file_path = f"emirates_ids/{os.path.basename(image_path)}"
            s3_uri = self.upload_to_s3(image_path, bucket_name, s3_file_path)

            # Get the object from S3
            response = self.textract_client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': s3_file_path
                    }
                }
            )

            # Extract text from all blocks
            extracted_text = " ".join([
                block['Text'] for block in response['Blocks']
                if block['BlockType'] == 'LINE'
            ])

            # Process text with LLM
            extracted_info = self.process_and_query(extracted_text)

            # Clean up S3
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_file_path)

            return extracted_info

        except Exception as e:
            raise Exception(f"Error processing Emirates ID: {str(e)}")

def display_results(extracted_info):
    """Display results in a formatted HTML table"""
    html = """
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #2c3e50; margin-bottom: 15px;">Extracted Information</h3>
        <table style="width: 100%; border-collapse: collapse;">
    """

    # Define field order and labels
    fields = [
        ('name', 'Name'),
        ('uid_no', 'UID Number'),
        ('passport_no', 'Passport Number'),
        ('profession', 'Profession'),
        ('sponsor', 'Sponsor'),
        ('place_of_issue', 'Place of Issue'),
        ('issue_date', 'Issue Date'),
        ('expiry_date', 'Expiry Date')
    ]

    for field, label in fields:
        value = extracted_info.get(field, 'Not Found')
        color = "#c0392b" if value == "Not Found" else "#2c3e50"
        html += f"""
            <tr style="border-bottom: 1px solid #ddd;">
                <td style="padding: 10px; font-weight: bold; color: #34495e; width: 30%;">{label}</td>
                <td style="padding: 10px; color: {color};">{value}</td>
            </tr>
        """

    html += """
        </table>
    </div>
    """
    display(HTML(html))

# The main() function has been removed since it's not needed in the module file
# All credential handling is now done in the Streamlit app