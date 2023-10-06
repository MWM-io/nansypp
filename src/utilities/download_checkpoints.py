import os
import zipfile

import pyrootutils
import requests

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_backbone(destination_dir):
    print("Downloading backbone..")
    file_id = "1ULSjTnK4wJYFiXzFCPbaZunKu9XGVb0u"
    tmp_dir = os.path.join(root, "static/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    destination_zip = os.path.join(tmp_dir, "backbone_hifitts.zip")
    download_file_from_google_drive(file_id, destination_zip)
    print("Unzipping...")
    os.makedirs(destination_dir, exist_ok=True)
    with zipfile.ZipFile(destination_zip, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    os.remove(destination_zip)


def download_tts(destination_dir):
    print("Downloading TTS...")
    file_id = "1oECzHoWyT98Ef9wJeCwnz77koLPhWZf6"
    tmp_dir = os.path.join(root, "static/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    destination_zip = os.path.join(tmp_dir, "tts_hifitts.zip")
    download_file_from_google_drive(file_id, destination_zip)
    print("Unzipping...")
    os.makedirs(destination_dir, exist_ok=True)
    with zipfile.ZipFile(destination_zip, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    os.remove(destination_zip)


def main():
    backbone_destination_dir = os.path.join(root, "static/runs/runs_backbone/hifitts")
    if not os.path.isdir(backbone_destination_dir):
        download_backbone(backbone_destination_dir)
    tts_destination_dir = os.path.join(root, "static/runs/runs_tts/hifitts")
    if not os.path.isdir(tts_destination_dir):
        download_tts(tts_destination_dir)


if __name__ == "__main__":
    main()
