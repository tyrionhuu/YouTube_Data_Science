import json
import urllib.parse as p

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

DEVELOPER_KEY = 'AIzaSyAiTWskBEIJVFrneK99_afD97mrff6yah0'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


# CHANNEL_DISPLAY_NAMES = ['FoxNews', 'CNN', 'BBCNews', 'ABCNews', 'NBCNews']

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
    # transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    # transcript_text = ""
    # for transcript in transcript_list:
    #     for item in transcript.fetch():
    #         transcript_text += item['text'] + " "
    # return transcript_text.strip()
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ""
        for item in transcript_list:
            transcript_text += item['text'] + " "
        return transcript_text.strip()
    except Exception as e:
        print("An error occurred:", e)
        return ""


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


def main():
    # Create a YouTube API client
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Get video details
    video_id = get_video_id_by_url("https://www.youtube.com/watch?v=S28SAhhA4XU")
    video_details = get_video_details(youtube, id=video_id)
    video_comments = get_video_comments(youtube, video_id)
    video_transcript = get_video_transcript(video_id)
    video_channel_display_name = video_details['items'][0]['snippet']['channelTitle']
    video_title = video_details['items'][0]['snippet']['title']
    # Save the video details to a JSON file
    video_info = {
        "video_details": video_details,
        "video_transcript": video_transcript,
        "video_comments": video_comments
    }

    with open(f"../data3/{video_channel_display_name}-{video_title}.json", "w") as json_file:
        json.dump(video_info, json_file, indent=4)

    print("Video details saved to 'video_info.json' file.")


if __name__ == '__main__':
    main()
