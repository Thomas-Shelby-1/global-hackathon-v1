from youtube_transcript_api import YouTubeTranscriptApi
tid = "VIDEO_ID_HERE"
tx = YouTubeTranscriptApi.get_transcript(tid, languages=["en"])
full = " ".join([t["text"] for t in tx])
print(full[:400], "...")
