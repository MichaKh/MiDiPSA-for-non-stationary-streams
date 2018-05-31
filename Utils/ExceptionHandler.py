import functools
import inspect
import logging
import traceback
import os


class ExceptionHandler(object):
    """
    Class implementing an exception handler and initiates a corresponding message based on the type of the exception.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    @staticmethod
    def halt_message(func, e):
        """
        Exception message formatting.
        Program exists with error code 1.
        :param func: Function which threw the exception
        :param e: Exception error
        :return: Return
        """
        ExceptionHandler.message(func, e)
        ExceptionHandler.logger.critical("Program aborts!")

        # Shutdown all loggers and flush their handlers.
        # Save the log from current run to the designated direcotry.
        logging.shutdown()
        while logging.getLogger("").handlers:
            logging.getLogger("").handlers.pop()
        # os.remove('Run_Log.log')
        exit(1)

    @staticmethod
    def writer_message(func, e):
        """
        Exception message formatting.
        Program notifies about the exception, after process completed.
        :param func: Function which threw the exception
        :param e: Exception error
        :return: Return
        """
        ExceptionHandler.logger.warning("Anonymization completed, but no output could be written to file! "
                                        "\nNo post-analysis of anonymization could be performed!")
        ExceptionHandler.message(func, e)
        return False

    @staticmethod
    def evaluation_message(func, e):
        """
        Exception message formatting.
        Warn about an evaluation metric that could not be calculated
        :param func: Function which threw the exception
        :param e: Exception error
        :return: Return
        """
        frm = inspect.trace()[-1]
        mod = inspect.getmodule(frm[0])
        modname = mod.__name__ if mod else frm[1]
        ExceptionHandler.logger.warning("Could not calculate evaluation metric: %s" % modname)
        ExceptionHandler.message(func, e)
        return False

    @staticmethod
    def external_lib_message(func, e):
        """
        Exception message formatting, caused by external library processes (e.g., MOA / WEKA).
        Program notifies about the exception, after process completed.
        :param func: Function which threw the exception
        :param e: Exception error
        :return: Return
        """
        ExceptionHandler.logger.warning("Attempted to preform post-analysis of anonymization, but subprocess failed!")
        ExceptionHandler.message(func, e)
        return False

    @staticmethod
    def message(func, e):
        """
        Exception message formatting.
        Prints function name, error type and error message, according to traceback
        :param func: Function which threw the exception
        :param e: Exception error
        :return: Return
        """
        # traceback.print_exc()
        frm = inspect.trace()[-1]
        mod = inspect.getmodule(frm[0])
        type_e = type(e).__name__
        modname = mod.__name__ if mod else frm[1]
        ExceptionHandler.logger.error("\nEXCEPTION: %s IN MODULE %s THROWN FROM FUNCTION: %s \nERROR: %s"
                                          % (type_e, modname, func.__name__, str(e)))
        # if mod:
        #     ExceptionHandler.print_stack_vars()

    # @staticmethod
    # def IO_message(func, e, dir_path, file_path):
    #     print "\nException", type(e).__name__, "in module:", module.__name__, 'in function:', func.__name__
    #     print "\nSource directory:", dir_path, "--> File:", file_path
    #     print "ERROR: " + str(e)
    #     ExceptionHandler.print_stack_vars()

    @staticmethod
    def print_stack_vars():
        """
        Print local variables in function which threw the exception, for further analysis.
        :return: Stack local variables
        """
        print("--------------------------------------------")
        print "{:<15} | {:<10}".format('Variable', 'Value')
        print("--------------------------------------------")
        for k, v in inspect.trace()[-1][0].f_locals.iteritems():
            print "{:<15} | {:<10}".format(k, v)

    @staticmethod
    def handle_exception(handler, *exceptions):
        """
        Function handler and wrapper for catching exceptions
        :param handler: Exception handler
        :param exceptions: Exception types to catch, or Exception (if types are not specified explicitly)
        :return: Return
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions or Exception as e:
                    return handler(func, e)
            return wrapper
        return decorator
