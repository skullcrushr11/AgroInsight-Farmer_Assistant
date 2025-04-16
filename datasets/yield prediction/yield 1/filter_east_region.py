import pandas as pd
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def filter_east_region():
    try:
        # Read the CSV file in chunks to handle large file size
        chunk_size = 100000  # Adjust based on your available memory
        chunks = pd.read_csv('crop_yield.csv', chunksize=chunk_size)
        
        # Initialize an empty list to store filtered chunks
        filtered_chunks = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}")
            
            # Filter for East region
            east_chunk = chunk[chunk['Region'] == 'East']
            
            if not east_chunk.empty:
                filtered_chunks.append(east_chunk)
        
        # Combine all filtered chunks
        if filtered_chunks:
            east_data = pd.concat(filtered_chunks, ignore_index=True)
            
            # Save the filtered data
            output_file = 'crop_yield_east.csv'
            east_data.to_csv(output_file, index=False)
            
            logger.info(f"Successfully filtered and saved {len(east_data)} records to {output_file}")
            logger.info(f"Original columns: {east_data.columns.tolist()}")
            logger.info(f"Sample of filtered data:\n{east_data.head()}")
        else:
            logger.warning("No data found for East region")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    filter_east_region() 