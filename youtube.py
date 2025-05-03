import os
import re
import ast
import asyncio
from yt_dlp import YoutubeDL

VIDEO_FILE = "chrisw.txt"
OUT_DIR    = "chrisw"

YTDL_ID_REGEX = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?.*v=|shorts/))([^?&\"'>]+)")

SEM = asyncio.Semaphore(os.cpu_count() or 4)

def extract_video_id(url: str) -> str:
    m = YTDL_ID_REGEX.search(url)
    if not m:
        raise ValueError(f"Could not extract video ID from: {url}")
    return m.group(1)

def download_audio_mp3(url: str, out_dir: str):
    """
    Synchronous function invoking yt-dlp to:
      • download best audio
      • convert to mp3 via ffmpeg
      • save under out_dir/<video_id>.mp3
    """
    vid = extract_video_id(url)
    out_tmpl = os.path.join(out_dir, f"{vid}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        # suppress console noise unless error
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

async def process_url(url: str, out_dir: str):
    """
    Wrapper to run download_audio_mp3 in a thread, under a semaphore.
    """
    async with SEM:
        loop = asyncio.get_running_loop()
        print(f"→ downloading audio for {url}")
        # run sync download in threadpool
        await loop.run_in_executor(None, download_audio_mp3, url, out_dir)
        print(f"✔ saved {extract_video_id(url)}.mp3")

async def main():
    raw = open(VIDEO_FILE).read()
    urls = ast.literal_eval(raw)
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [process_url(u, OUT_DIR) for u in urls]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
