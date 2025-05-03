import re, os, time, aiohttp, asyncio, traceback
from typing import List
import aiofiles
import ast

API_KEY  = "6fae47ef06msh54dcb697486c67ep17f9cfjsn54d2a7fd19df"
API_HOST = "youtube-to-mp315.p.rapidapi.com"
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

async def get_mp3_link(session: aiohttp.ClientSession, video_url: str, max_retries=8, delay=4) -> str:
    # 1) Kick off conversion
    print(f"[DEBUG] POST {DOWNLOAD_ENDPOINT}?url={video_url}")
    async with session.post(DOWNLOAD_ENDPOINT, headers=HEADERS, params={"url": video_url}) as resp:
        resp.raise_for_status()
        job = await resp.json()
        job_id = job["id"]
        print(f"[DEBUG] Got job id = {job_id}, initial status = {job.get('status')}")

    # 2) Poll the status endpoint until it's no longer CONVERTING
    status_url = f"{STATUS_ENDPOINT}/{job_id}"
    for i in range(1, max_retries+1):
        print(f"[DEBUG] GET {status_url} (attempt {i})")
        async with session.get(status_url, headers=HEADERS) as r2:
            r2.raise_for_status()
            s = await r2.json()
            st = s.get("status","<no-status>")
            print(f"[DEBUG] status = {st!r}")

            if st.upper() != "CONVERTING" and "downloadUrl" in s:
                print(f"[DEBUG] Conversion finished; downloadUrl = {s['downloadUrl']!r}")
                return s["downloadUrl"]

        await asyncio.sleep(delay)

    raise TimeoutError(f"job {job_id} still not ready after {max_retries} tries (last status={st!r})")

async def download_file(session: aiohttp.ClientSession, url: str, dest: str):
    print(f"[DEBUG] downloading file from {url}")
    async with session.get(url) as r:
        r.raise_for_status()
        async with aiofiles.open(dest, "wb") as f:
            async for chunk in r.content.iter_chunked(8192):
                await f.write(chunk)

async def process_url(session: aiohttp.ClientSession, url: str, out_dir: str) -> bool:
    try:
        vid = extract_video_id(url)
        if f'{vid}.mp3' in os.listdir(out_dir):
            print(f"Skipping {url} because it already exists")
            return True
            
        print(f"Processing video ID: {vid}")
        watch = f"https://www.youtube.com/watch?v={vid}"
        mp3_url = await get_mp3_link(session, watch)
        out_path = os.path.join(out_dir, f"{vid}.mp3")
        await download_file(session, mp3_url, out_path)
        print("✅ done\n")
        return True
    except Exception as e:
        print(f"❌ Failed for {url}: {e}")
        traceback.print_exc()
        print()
        return False

async def process_urls(urls: List[str], out_dir: str = "mp3_downloads", max_concurrent: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    
    # Create a semaphore to limit concurrent downloads
    sem = asyncio.Semaphore(max_concurrent)
    
    async def bounded_process(url: str) -> bool:
        async with sem:
            async with aiohttp.ClientSession() as session:
                return await process_url(session, url, out_dir)
    
    # Process URLs concurrently with bounded concurrency
    tasks = [bounded_process(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Print summary
    successful = sum(1 for r in results if r is True)
    print(f"\nProcessing complete: {successful}/{len(urls)} successful")

async def main(urls: List[str], out_dir: str):
    print(f"Processing {len(urls)} urls")
    await process_urls(urls, out_dir)

if __name__ == "__main__":
    video_file = "chrisw.txt"
    output_dir = "chrisw"

    urls = open(video_file).read()
    urls = ast.literal_eval(urls)
    
    # Run the async main function
    asyncio.run(main(urls, output_dir))