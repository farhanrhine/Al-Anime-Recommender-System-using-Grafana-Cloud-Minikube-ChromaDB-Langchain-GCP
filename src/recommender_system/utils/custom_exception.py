import traceback
import sys

class CustomException(Exception):
    """
    Custom exception for detailed error reporting in the project.
    Captures:
    - original message
    - root cause error
    - filename and line number
    """

    def __init__(self, message: str, error: Exception = None):
        detailed_message = self._build_detailed_message(message, error)
        super().__init__(detailed_message)
        self.detailed_message = detailed_message

    @staticmethod
    def _build_detailed_message(message: str, error: Exception = None) -> str:
        exc_type, exc_value, exc_tb = sys.exc_info()

        # When error didn't come from inside an exception block
        if exc_tb is None:
            file_name = "Unknown File"
            line_number = "Unknown Line"
        else:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno

        root_cause = f"{error}" if error else "No additional error detail"

        return (
            f"{message}\n"
            f"Root Cause: {root_cause}\n"
            f"File: {file_name}\n"
            f"Line: {line_number}"
        )

    def __str__(self): # beautiful message instead of Python's ugly default one.
        return self.detailed_message
