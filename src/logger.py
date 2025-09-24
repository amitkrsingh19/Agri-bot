import logging
import os
from datetime import datetime

logs="logs_directory"
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(LOG_PATH,exist_ok=True)
log_file_path = os.path.join(LOG_PATH,LOG_FILE)

# Configure basic logging
logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s] %(lineno)d  %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename=log_file_path)
logger = logging.getLogger(__name__)

