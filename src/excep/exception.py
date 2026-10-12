import sys

def error_message_details(error_message, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    linenumber = exc_tb.tb_lineno
    
    error_message_str = "Error occurred in python script named [{0}] line number [{1}] with message [{2}]".format(
        filename, linenumber, str(error_message)
    )
    return error_message_str

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        
        self.error_message = error_message_details(
            error_message, error_details=error_details
        )
       
   
    def __str__(self):
        return self.error_message
