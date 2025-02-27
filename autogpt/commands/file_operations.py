"""File operations for AutoGPT"""
from __future__ import annotations

import os
import os.path
from typing import Generator

import requests
from colorama import Back, Fore
from requests.adapters import HTTPAdapter, Retry

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.spinner import Spinner
from autogpt.utils import readable_file_size

CFG = Config()


def check_duplicate_operation(operation: str, filename: str) -> bool:
    """Check if the operation has already been performed on the given file

    Args:
        operation (str): The operation to check for
        filename (str): The name of the file to check for

    Returns:
        bool: True if the operation has already been performed on the file
    """
    log_content = read_file(CFG.file_logger_path)
    log_entry = f"{operation}: {filename}\n"
    return log_entry in log_content


def log_operation(operation: str, filename: str) -> None:
    """Log the file operation to the file_logger.txt

    Args:
        operation (str): The operation to log
        filename (str): The name of the file the operation was performed on
    """
    log_entry = f"{operation}: {filename}\n"
    append_to_file(CFG.file_logger_path, log_entry, should_log=False)


def split_file(
    content: str, max_length: int = 4000, overlap: int = 0
) -> Generator[str, None, None]:
    """
    Split text into chunks of a specified maximum length with a specified overlap
    between chunks.

    :param content: The input text to be split into chunks
    :param max_length: The maximum length of each chunk,
        default is 4000 (about 1k token)
    :param overlap: The number of overlapping characters between chunks,
        default is no overlap
    :return: A generator yielding chunks of text
    """
    start = 0
    content_length = len(content)

    while start < content_length:
        end = start + max_length
        if end + overlap < content_length:
            chunk = content[start : end + overlap - 1]
        else:
            chunk = content[start:content_length]

            # Account for the case where the last chunk is shorter than the overlap, so it has already been consumed
            if len(chunk) <= overlap:
                break

        yield chunk
        start += max_length - overlap


@command("read_file", "Read file", '"filename": "<filename>"')
def read_file(filename: str) -> str:
    """Read a file and return the contents

    Args:
        filename (str): The name of the file to read

    Returns:
        str: The contents of the file
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error: {str(e)}"


def ingest_file(
    filename: str, memory, max_length: int = 4000, overlap: int = 200
) -> None:
    """
    Ingest a file by reading its content, splitting it into chunks with a specified
    maximum length and overlap, and adding the chunks to the memory storage.

    :param filename: The name of the file to ingest
    :param memory: An object with an add() method to store the chunks in memory
    :param max_length: The maximum length of each chunk, default is 4000
    :param overlap: The number of overlapping characters between chunks, default is 200
    """
    try:
        print(f"Working with file {filename}")
        content = read_file(filename)
        content_length = len(content)
        print(f"File length: {content_length} characters")

        chunks = list(split_file(content, max_length=max_length, overlap=overlap))

        num_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            print(f"Ingesting chunk {i + 1} / {num_chunks} into memory")
            memory_to_add = (
                f"Filename: {filename}\n" f"Content part#{i + 1}/{num_chunks}: {chunk}"
            )

            memory.add(memory_to_add)

        print(f"Done ingesting {num_chunks} chunks from {filename}.")
    except Exception as e:
        print(f"Error while ingesting file '{filename}': {str(e)}")


@command("write_to_file", "Write to file", '"filename": "<filename>", "text": "<text>"')
def write_to_file(filename: str, text: str) -> str:
    """Write text to a file

    Args:
        filename (str): The name of the file to write to
        text (str): The text to write to the file

    Returns:
        str: A message indicating success or failure
    """
    try:
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        log_operation("write", filename)
        return "File written to successfully."
    except Exception as e:
        return f"Error: {str(e)}"


@command(
    "append_to_file", "Append to file", '"filename": "<filename>", "text": "<text>"'
)
def append_to_file(filename: str, text: str, should_log: bool = True) -> str:
    """Append text to a file

    Args:
        filename (str): The name of the file to append to
        text (str): The text to append to the file
        should_log (bool): Should log output

    Returns:
        str: A message indicating success or failure
    """
    try:
        with open(filename, "a") as f:
            f.write(text)

        if should_log:
            log_operation("append", filename)

        return "Text appended successfully."
    except Exception as e:
        return f"Error: {str(e)}"


@command("delete_file", "Delete file", '"filename": "<filename>"')
def delete_file(filename: str) -> str:
    """Delete a file

    Args:
        filename (str): The name of the file to delete

    Returns:
        str: A message indicating success or failure
    """
    try:
        os.remove(filename)
        log_operation("delete", filename)
        return "File deleted successfully."
    except Exception as e:
        return f"Error: {str(e)}"


@command("search_files", "Search Files", '"directory": "<directory>"')
def search_files(directory: str) -> list[str]:
    """Search for files in a directory

    Args:
        directory (str): The directory to search in

    Returns:
        list[str]: A list of files found in the directory
    """
    found_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("."):
                continue
            relative_path = os.path.relpath(
                os.path.join(root, file), CFG.workspace_path
            )
            found_files.append(relative_path)

    return found_files


@command(
    "download_file",
    "Download File",
    '"url": "<url>", "filename": "<filename>"',
    CFG.allow_downloads,
    "Error: You do not have user authorization to download files locally.",
)
def download_file(url, filename):
    """Downloads a file
    Args:
        url (str): URL of the file to download
        filename (str): Filename to save the file as
    """
    try:
        message = f"{Fore.YELLOW}Downloading file from {Back.LIGHTBLUE_EX}{url}{Back.RESET}{Fore.RESET}"
        with Spinner(message) as spinner:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            total_size = 0
            downloaded_size = 0

            with session.get(url, allow_redirects=True, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("Content-Length", 0))
                downloaded_size = 0

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Update the progress message
                        progress = f"{readable_file_size(downloaded_size)} / {readable_file_size(total_size)}"
                        spinner.update_message(f"{message} {progress}")

            return f'Successfully downloaded and locally stored file: "{filename}"! (Size: {readable_file_size(total_size)})'
    except requests.HTTPError as e:
        return f"Got an HTTP Error whilst trying to download file: {e}"
    except Exception as e:
        return "Error: " + str(e)
