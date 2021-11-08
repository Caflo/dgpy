from datetime import datetime
import platform
import os

class Generation_Details():

    """
    Useful generation details
    start_ts: timestamp of generation at start
    end_ts: timestamp of generation at end

    When invoked constructor, there will be stored 2 more attributes:
    platform_name: OS name
    user: who logged in the OS
    """

    def __init__(self, start_ts, end_ts): # user is O.S user, i.e. refers to the name who called the process
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.platform_name = platform.platform()

        if self.platform_name.startswith("Windows"):
            user = os.environ.get('USERNAME')
        if self.platform_name.startswith("linux") or self.platform_name.startswith("darwin"):
            user = os.environ.get('USER')

        self.user = user
