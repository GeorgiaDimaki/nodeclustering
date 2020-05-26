import datetime as dt


class Logger(object):
    """Simple logger class.

    Attributes:
        log_file (str): the path to the logfile to manage
    """

    def __init__(self, filename):
        """Convert timestamp to h:mm:ss string.

         Args:
            filename (str): the path to the logfile to manage

        Returns:
            no value
        """
        self.log_file = filename

    def timestamp(self):
        """Records a timestamp of current date and time in the logfile and the command line.

        Returns:
            no value
        """
        with open(self.log_file, "a") as f:
            f.write('[TIMESTAMP] {}\n'.format(str(dt.datetime.now())))
        print(dt.datetime.now())

    def log(self, s, time=False):
        """Logs the string s in the logfile. If time is true it records a timestamp first.

        Returns:
            no value
        """
        if time:
            self.timestamp()
        with open(self.log_file, "a") as f:
            f.write(s+"\n")
