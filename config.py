import os


temp_objective_path = "objectives.txt"
temp_warning_path = "warning.txt"

WORKING_DIR = "3d_rec_drone"
OBJECTIVE_FILE = temp_objective_path
WARNINGS_FILE = temp_warning_path
DB_DIR = os.path.join(WORKING_DIR, "DB")

STORE_OBJECTIVE_FILE = os.path.join(WORKING_DIR, "objectives.txt")
STORE_WARNINGS_FILE = os.path.join(WORKING_DIR, "warnings.txt")
LONG_TERM_MEMORY_FILE = os.path.join(WORKING_DIR, "long_term_memory.txt")
STATUS_REPORT_FILE = os.path.join(WORKING_DIR, "status_reports.txt")
SUMMARY_FILE = os.path.join(WORKING_DIR, "summary.txt")