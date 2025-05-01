import re, os, time, requests, traceback

API_KEY  = "6fae47ef06msh54dcb697486c67ep17f9cfjsn54d2a7fd19df"
API_HOST = "coolguruji-youtube-to-mp3-download-v1.p.rapidapi.com"
DOWNLOAD_ENDPOINT = f"https://{API_HOST}/download"
STATUS_ENDPOINT   = f"https://{API_HOST}/status"     # we will append “/{id}”

HEADERS = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST,
}

YTDL_ID_REGEX = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?.*v=|shorts/))([^?&\"'>]+)"
)

def extract_video_id(url: str) -> str:
    m = YTDL_ID_REGEX.search(url)
    if not m:
        raise ValueError(f"Could not extract video ID from: {url}")
    return m.group(1)

def get_mp3_link(video_url: str, max_retries=8, delay=4) -> str:
    # 1) Kick off conversion
    print(f"[DEBUG] POST {DOWNLOAD_ENDPOINT}?url={video_url}")
    resp = requests.post(DOWNLOAD_ENDPOINT, headers=HEADERS, params={"url": video_url})
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]
    print(f"[DEBUG] Got job id = {job_id}, initial status = {job.get('status')}")

    # 2) Poll the status endpoint until it’s no longer CONVERTING
    status_url = f"{STATUS_ENDPOINT}/{job_id}"
    for i in range(1, max_retries+1):
        print(f"[DEBUG] GET {status_url} (attempt {i})")
        r2 = requests.get(status_url, headers=HEADERS)
        r2.raise_for_status()
        s = r2.json()
        st = s.get("status","<no-status>")
        print(f"[DEBUG] status = {st!r}")

        if st.upper() != "CONVERTING" and "downloadUrl" in s:
            print(f"[DEBUG] Conversion finished; downloadUrl = {s['downloadUrl']!r}")
            return s["downloadUrl"]

        time.sleep(delay)

    raise TimeoutError(f"job {job_id} still not ready after {max_retries} tries (last status={st!r})")

def download_file(url: str, dest: str):
    print(f"[DEBUG] downloading file from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def main(short_urls, out_dir="mp3_downloads"):
    os.makedirs(out_dir, exist_ok=True)
    for s in short_urls:
        try:
            vid = extract_video_id(s)
            print(f"Processing video ID: {vid}")
            watch = f"https://www.youtube.com/watch?v={vid}"
            mp3_url = get_mp3_link(watch)
            out_path = os.path.join(out_dir, f"{vid}.mp3")
            download_file(mp3_url, out_path)
            print("✅ done\n")
        except Exception as e:
            print(f"❌ Failed for {s}: {e}")
            traceback.print_exc()
            print()


urls = eval(open("shorts.txt").read())
print(f"Processing {len(urls)} urls")
main(urls)
