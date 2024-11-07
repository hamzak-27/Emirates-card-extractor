import streamlit as st
import json
import base64
from emirates_id_extractor import EmiratesIDExtractor
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Emirates ID Information Extractor",
    page_icon="🆔",
    layout="wide"
)

# Custom CSS for styling - keeping the same dark theme
st.markdown("""
<style>
    /* General styling */
    .stApp {
        background-color: #1a1a1a;
    }
    
    /* Description box styling */
    .description {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #ffffff90;
    }
    
    /* Process button hover effect */
    .stButton > button {
        width: 100%;
        background-color: #0e76a8 !important;
        color: white !important;
        transition: background-color 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #0a5a8a !important;
        border-color: #0a5a8a !important;
    }
    
    /* Output container styling */
    .output-container {
        background-color: rgb(17, 23, 29);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .field-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .field-value {
        color: white;
        font-size: 14px;
        margin-bottom: 15px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Emirates ID Information Extractor")
    
    # Tool description
    st.markdown("""
        <div class="description">
            This tool extracts information from Emirates ID cards using advanced AI processing. 
            Upload your Emirates ID image and click 'Process the Card' to get the extracted information 
            in a clean, formatted output. All data is processed securely and not stored.
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize credentials from secrets
    AWS_REGION = st.secrets["aws"]["region"]
    AWS_ACCESS_KEY = st.secrets["aws"]["access_key"]
    AWS_SECRET_KEY = st.secrets["aws"]["secret_key"]
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
    BUCKET_NAME = st.secrets["aws"]["bucket_name"]

    # Cache the extractor instance
    @st.cache_resource
    def get_extractor():
        return EmiratesIDExtractor(
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            openai_api_key=OPENAI_API_KEY
        )

    extractor = get_extractor()

    # Initialize session state for tracking current file
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

    # File uploader - updated to accept image formats
    uploaded_file = st.file_uploader("Upload Emirates ID", type=['jpg', 'jpeg', 'png', 'pdf'])

    # Clear results if a new file is uploaded
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if (st.session_state.current_file_name != current_file_name):
            st.session_state.current_file_name = current_file_name
            if 'extracted_info' in st.session_state:
                del st.session_state.extracted_info
            if 'file_content' in st.session_state:
                del st.session_state.file_content
        
        # Store new file content
        if 'file_content' not in st.session_state:
            st.session_state.file_content = uploaded_file.read()

        if st.button("Process the Card", type="primary"):
            with st.spinner("Processing Emirates ID..."):
                try:
                    # Create temporary file
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(st.session_state.file_content)
                        temp_path = tmp_file.name

                    # Process the file
                    extracted_info = extractor.extract_text_from_image(temp_path, BUCKET_NAME)
                    os.unlink(temp_path)  # Clean up temporary file

                    # Store results in session state
                    st.session_state.extracted_info = extracted_info

                except Exception as e:
                    st.error(f"Error processing Emirates ID: {str(e)}")

        # Display results if available
        if hasattr(st.session_state, 'extracted_info'):
            # Create columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="field-label">Name</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("name", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">UID Number</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("uid_no", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">Passport Number</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("passport_no", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">Profession</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("profession", "Not Found")}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="field-label">Sponsor</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("sponsor", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">Place of Issue</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("place_of_issue", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">Issue Date</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("issue_date", "Not Found")}</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="field-label">Expiry Date</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="field-value">{st.session_state.extracted_info.get("expiry_date", "Not Found")}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()