import sys



def error_message_details(error_message,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    linenumber=exc_tb.tb_lineno
    error=str(error)
    
    error_message="error occured in python script named [{0}] line number [{1}]  with message [{2}]".format(filename,linenumber,error)
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details: sys):
        
       super().__init__(error_message)
       
       
       self.error_message=error_message_details(
           error_message , error_detail=error_details
       )
       
       def __str__(self):
           self.error_message