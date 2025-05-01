import re
import os
import time
import ast
import requests
import traceback

# — RapidAPI credentials & base URL —
API_KEY  = "6fae47ef06msh54dcb697486c67ep17f9cfjsn54d2a7fd19df"
API_HOST = "coolguruji-youtube-to-mp3-download-v1.p.rapidapi.com"
BASE_URL = f"https://{API_HOST}"    # we will GET "/" with ?id=<video_id> :contentReference[oaicite:0]{index=0}

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST,
}

# — regex to pull the 11-char video ID from any YouTube URL (watch/shorts/youtu.be) —
YTDL_ID_REGEX = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?.*v=|shorts/))([^?&\"'>]+)"
)

def extract_video_id(url: str) -> str:
    m = YTDL_ID_REGEX.search(url)
    if not m:
        raise ValueError(f"Could not extract video ID from: {url}")
    return m.group(1)

def get_mp3_link(video_id: str, max_retries=5, delay=3) -> str:
    """
    Call GET /?id=<video_id>.
    The JSON response contains 'downloadUrl' once conversion is done.
    """
    params = {"id": video_id}
    for attempt in range(1, max_retries+1):
        print(f"[DEBUG] GET {BASE_URL}/?id={video_id}  (attempt {attempt})")
        resp = requests.get(BASE_URL+"/", headers=HEADERS, params=params)
        # if we hit rate-limit or other HTTP error, show body and retry/raise
        if not resp.ok:
            print(f"[DEBUG] HTTP {resp.status_code} response:\n{resp.text}")
            resp.raise_for_status()

        # parse JSON
        data = resp.json()
        status = data.get("status","<no-status>")
        print(f"[DEBUG] status = {status!r}, data keys = {list(data.keys())}")

        # once 'downloadUrl' appears, return it
        if "downloadUrl" in data:
            return data["downloadUrl"]

        # otherwise wait and retry
        time.sleep(delay)

    raise TimeoutError(f"Video {video_id} not ready after {max_retries} tries; last status={status!r}")

def download_file(url: str, dest: str):
    print(f"[DEBUG] downloading file from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def main(urls, out_dir="mp3_downloads"):
    os.makedirs(out_dir, exist_ok=True)
    for url in urls:
        try:
            vid = extract_video_id(url)
            print(f"Processing video ID: {vid}")
            mp3_url = get_mp3_link(vid)
            out_path = os.path.join(out_dir, f"{vid}.mp3")
            download_file(mp3_url, out_path)
            print("✅ done\n")
        except Exception as e:
            print(f"❌ Failed for {url}: {e}")
            traceback.print_exc()
            print()

if __name__ == "__main__":
    # read your list of shorts URLs from shorts.txt
    raw = open("shorts.txt").read()
    shorts = ast.literal_eval(raw)
    print(f"Processing {len(shorts)} URLs…")
    main(shorts)
