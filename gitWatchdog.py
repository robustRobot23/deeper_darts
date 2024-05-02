import time
from git import Repo

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

# Start monitoring the directory for changes
if __name__ == "__main__":
    while True:
        print("Checking for changes...")
        push_changes()
        time.sleep(1200)  # 1200 seconds = 20 minutes
