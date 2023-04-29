# This script is used to backup the modified files in the project folder.
import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class BackupHandler(FileSystemEventHandler):
    def __init__(self, target_folder):
        self.target_folder = target_folder

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".py"):
            # 忽略以 "backup_" 开头的文件夹
            if not "backup_" in event.src_path:
                self.backup_modified_file(event.src_path)

    def backup_modified_file(self, file_path):
        current_time = time.strftime("%Y%m%d-%H%M")
        filename, file_extension = os.path.splitext(os.path.basename(file_path))
        backup_file_name = f"{filename}_{current_time}{file_extension}"
        backup_file_path = os.path.join(self.target_folder, backup_file_name)
        shutil.copy2(file_path, backup_file_path)
        print(f"File {file_path} has been backed up in {backup_file_path}")


def main():
    project_folder = "../"
    target_folder = "../../CodeBackup/"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    event_handler = BackupHandler(target_folder)
    observer = Observer()
    observer.schedule(event_handler, project_folder, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()

