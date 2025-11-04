import io
import os
import sys
import logging
import time
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import numpy as np
from requests.exceptions import ConnectionError
from http.client import RemoteDisconnected

@contextmanager
def silence_all():
    # capture stdout/stderr
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # silence logging & warnings
        prev_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                logging.disable(prev_disable)

def retry_request(func, tries=10, delay=2):
    """Retry a network call a few times before failing."""
    for attempt in range(tries):
        try:
            return func()
        except (ConnectionError, RemoteDisconnected) as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            if attempt == tries - 1:
                raise
            time.sleep(delay)

if __name__ == "__main__":
    # If this script is run directly, silence all output
    with silence_all():
        print("This will not be printed.")

    print("This will be printed if run as a module, but not if run directly.")
    
