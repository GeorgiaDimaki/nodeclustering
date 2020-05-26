import datetime as dt
import time


class TimingManager(object):
    """Context Manager used with the statement 'with' to time some execution.

    Example:

    with TimingManager() as t:
       # Code to time


    Attributes:
        clock (func): a function that returns fractional seconds of system and user CPU time of current process
               .. _Documentation: https://docs.python.org/3.7/library/time.html#time.process_time
        filename (str): the path to the log file
        log_file (file): the log file to record the elapsed time of the process under timing
        start (float): the start - undefined reference point - of the process under timing
    """

    clock = time.process_time

    def __init__(self, filename):
        """Initialize Timing Manager.

        Args:
            filename (str): the name of the log file to register the time

        Returns:
            no value
        """
        self.filename = filename

    def __enter__(self):
        """Records - in object variable - the start time of process under timing.

        Returns:
            self (TimeManager)
        """
        self.log_file = open(self.filename, 'a')
        self.start = self.clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Logs in the log file the elapsed time of the process under timing.

        Returns:
            no value
        """
        self.endlog()
        self.log_file.close()

    def endlog(self):
        """Logs the elapsed time in the logfile

       Returns:
           no value
       """
        self.log_file.write('[E] {}\n'.format(self.elapsed()))

    def elapsed(self):
        """Return current elapsed time as hh:mm:ss string.

        Returns:
            str : string representation of elapsed time
        """
        return self._secondsToStr(sec=(self.clock() - self.start))

    def _secondsToStr(self, sec):
        """Convert timestamp to h:mm:ss string.

         Args:
            sec (Timestamp): the timestamp to convert

        Returns:
            str : string representation of sec
        """
        return str(dt.timedelta(seconds=sec))
