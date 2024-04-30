import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from git import Repo

# Define the directory to monitor
directory_to_watch = 'runs/detect/DeeperDarts'

# Define the Git repository path
repo_path = ''

# Function to push changes to the repository
def push_changes():
    try:
        repo = Repo(repo_path)
        repo.git.add('.')
        repo.index.commit("Automatic commit by watcher script")
        repo.git.push()
        print("Changes pushed to repository successfully")
    except Exception as e:
        print(f"Error pushing changes to repository: {e}")

# Define a class to handle file system events
class Watcher(FileSystemEventHandler):
    def on_any_event(self, event):
        print(f"Detected change: {event.event_type} - {event.src_path}")
        push_changes()

# Start monitoring the directory for changes
if __name__ == "__main__":
    while True:
        print("Checking for changes...")
        push_changes()
        time.sleep(300)  # 300 seconds = 5 minutes
