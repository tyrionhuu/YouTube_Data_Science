import json
import os
import time
import urllib.parse as p

import tqdm
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

DEVELOPER_KEY = 'AIzaSyAiTWskBEIJVFrneK99_afD97mrff6yah0'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
CHANNEL_DISPLAY_NAMES = ['FoxNews', 'CNN', 'BBCNews', 'ABCNews', 'NBCNews']


def get_channel_id_by_name(youtube, channel_display_name):
    """
    Return the Channel ID from the channel display name
    """
    response = youtube.channels().list(
        part="id",
        forUsername=channel_display_name
    ).execute()
    if 'items' in response and len(response['items']) > 0:
        return response['items'][0]['id']
    else:
        raise Exception(f"Channel with display name '{channel_display_name}' not found.")


def get_video_id_by_url(url):
    """
    Return the Video ID from the video `url`
    """
    # Split URL parts
    parsed_url = p.urlparse(url)
    # Get the video ID by parsing the query of the URL
    video_id = p.parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    else:
        raise Exception(f"Wasn't able to parse video URL: {url}")


def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()


def get_video_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript_text = ""
    for transcript in transcript_list:
        for item in transcript.fetch():
            transcript_text += item['text'] + " "
    return transcript_text.strip()


def get_comment_replies(youtube, comment_id):
    replies = []
    nextPageToken = None

    while True:
        try:
            response = youtube.comments().list(
                part="snippet",
                parentId=comment_id,
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                # Extract commenter's information
                commenter_display_name = item['snippet']['authorDisplayName']
                commenter_channel_id = item['snippet']['authorChannelId']['value']
                comment_text = item['snippet']['textDisplay']
                likes_count = item['snippet']['likeCount']

                # Create comment dictionary
                reply_data = {
                    "commenter_display_name": commenter_display_name,
                    "commenter_channel_id": commenter_channel_id,
                    "comment_text": comment_text,
                    "likes_count": likes_count
                }

                replies.append(reply_data)

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

        except Exception as e:
            print("An error occurred:", e)
            break

    return replies


def get_video_comments(youtube, video_id):
    comments = []
    nextPageToken = None

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response['items']:
                # Extract commenter's information
                commenter_display_name = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                commenter_channel_id = item['snippet']['topLevelComment']['snippet']['authorChannelId']['value']
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                likes_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                comment_id = item['snippet']['topLevelComment']['id']

                # Get replies for this comment
                comment_replies = get_comment_replies(youtube, comment_id)

                # Create comment dictionary
                comment_data = {
                    "commenter_display_name": commenter_display_name,
                    "commenter_channel_id": commenter_channel_id,
                    "comment_text": comment_text,
                    "likes_count": likes_count,
                    "replies": comment_replies
                }

                comments.append(comment_data)

            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

        except Exception as e:
            print("An error occurred:", e)
            break

    return comments


def fetch_video_info(youtube, channel_display_name):
    # Get the channel ID based on the display name
    channel_id = get_channel_id_by_name(youtube, channel_display_name)

    # Get the latest video from the channel
    latest_video = get_latest_video(youtube, channel_id)

    # Extract video ID from the latest video
    video_id = latest_video['items'][0]['id']['videoId']

    # Fetch video details, transcript, and comments
    video_info = {
        "video_id": video_id,
        "video_details": get_video_details(youtube, id=video_id),
        "video_transcript": get_video_transcript(video_id),
        "video_comments": get_video_comments(youtube, video_id)
    }

    return video_info


def get_latest_video(youtube, channel_id):
    # Get the latest video from the channel
    response = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=1,
        order="date"
    ).execute()

    return response


def main():
    # Build the service
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Output directory
    output_dir = "../video_info"
    # Create a directory to store video information
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        for channel_display_name in CHANNEL_DISPLAY_NAMES:
            # Fetch video info for each channel
            video_info = fetch_video_info(youtube, channel_display_name)

            # File name
            file_name = f"{output_dir}/{video_info['video_id']}.json"

            # Save video info as JSON
            with open(file_name, 'w') as json_file:
                json.dump(video_info, json_file, indent=4)

        # Sleep
        print("Waiting for 1 min...")
        for _ in tqdm.tqdm(range(60)):
            time.sleep(1)


if __name__ == '__main__':
    main()
