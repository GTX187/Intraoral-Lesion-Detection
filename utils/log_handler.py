import logging
from logging.handlers import TimedRotatingFileHandler
import os


class TransactionIDFilter(logging.Filter):
    """
    Ensures every log record has a transaction_id attribute for the formatter.
    """

    def filter(self, record):
        if not hasattr(record, "transaction_id"):
            record.transaction_id = "-"  # or ""
        return True


class CustomSizeDayRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(
        self,
        filename,
        when="h",
        backupCount=0,
        size=4,  # Size in MB
        interval=1,
        encoding=None,
        delay=False,
        utc=False,
        atTime=None,
    ):
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime,
        )
        self.size = size * 1024 * 1024
        self.addFilter(TransactionIDFilter())

    def shouldRollover(self, record):
        """
        Return True if rollover should occur.
        This happens if either:
          - The file size exceeds 4MB, or
          - It's time to do the time-based rollover (e.g., at midnight).
        """
        # Check size condition
        if self.stream is None:  # Just in case
            self.stream = self._open()
        self.stream.flush()
        if os.stat(self.baseFilename).st_size >= self.size:
            return True

        # Check time condition (same as base class)
        t = int(self.rolloverAt)
        if int(record.created) >= t:
            return True
        return False


class TransactionLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # Add the transaction_id to every log message
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["transaction_id"] = self.extra["transaction_id"]
        return msg, kwargs
