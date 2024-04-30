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
    observer = Observer()
    observer.schedule(Watcher(), path=directory_to_watch, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
