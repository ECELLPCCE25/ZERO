import logging
from fastapi import APIRouter, HTTPException, Query
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter()

@router.get("/get_person_count_by_stream_id")
def get_person_count_by_stream_id(user_id: str = Query(...), stream_id: str = Query(...)):
    try:
        # Query the Supabase database to retrieve person count data
        response = supabase.table('cam_data').select('person_count_data').eq('user_id', user_id).eq('stream_id', stream_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="No data found for the specified user and stream ID")

        return response.data[0]['person_count_data']
    except Exception as e:
        logging.error(f"Error retrieving person count data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@router.get("/get_streams_by_user_id")
def get_person_count_by_stream_id(user_id: str = Query(...)):
    try:
        # Query the Supabase database to retrieve person count data
        response = supabase.table('cam_data').select('stream_id').eq('user_id', user_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="No data found for the specified user and stream ID")

        return response.data
    except Exception as e:
        logging.error(f"Error retrieving person count data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")