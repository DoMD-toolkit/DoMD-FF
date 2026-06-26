import os

from misc.logger import logger
from opls.opls_db.database import OplsDB

this_dir, this_file = os.path.split(__file__)
logger.info(f"Loading {os.path.join(this_dir, 'resources', 'opls.db')}")
opls_db = OplsDB(os.path.join(this_dir, 'resources', 'opls.db'))
