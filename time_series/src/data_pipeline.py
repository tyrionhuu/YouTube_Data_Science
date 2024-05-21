import os

from googleapiclient.discovery import build
from datetime import datetime, timedelta
import csv


DEVELOPER_KEY = 'AIzaSyAiTWskBEIJVFrneK99_afD97mrff6yah0'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
HISTORY_VIDEO_DIR = '../videos/history'
CHANNEL_IDS = {
    'ABCNews': 'UCBi2mrWuNuyYy4gbM6fU18Q',
    'FoxNews': 'UCXIJgqnII2ZOINSWNOGFThA',
    'CNN': 'UCupvZG-5ko_eiXAupbDfxWw',
    'NBCNews': 'UCeY0bbntWzzVIaj2z3QigXg',
    'BBCNews': 'UC16niRr50-MSBwiO3YDb3RA',
    'aljazeeraenglish': 'UCNye-wNBqNL5ZzHSJj3l8Bg'
}

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)


def get_video_statistics(video_id):
    # Call the API to retrieve video statistics
    request = youtube.videos().list(
        part='statistics',
        id=video_id
    )
    response = request.execute()
    # Extract statistics from the response
    statistics = response['items'][0]['statistics']
    return statistics


# Function to retrieve videos from a channel 14 days ago and save them to a CSV file
def get_channel_videos_14_days_ago(channel_id):
    # Get the date 14 days ago
    date_14_days_ago = datetime.now() - timedelta(days=14)
    date_str = date_14_days_ago.strftime('%Y-%m-%d')
    # date_str = '2024-05-07'

    # Call the API to retrieve videos from the channel 14 days ago
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        maxResults=50,
        order='date',
        type='video',
        publishedAfter=f'{date_str}T00:00:00Z',
        publishedBefore=f'{date_str}T23:59:59Z'
    )
    response = request.execute()

    # Process the videos
    videos = []
    for item in response.get('items', []):
        # Get video statistics
        video_id = item['id']['videoId']
        statistics = get_video_statistics(video_id)
        video_info = {
            'channel_name': item['snippet']['channelTitle'],
            'video_title': item['snippet']['title'],
            'published_at': item['snippet']['publishedAt'],
            'video_id': item['id']['videoId'],
            'view_count': statistics['viewCount'] if 'viewCount' in statistics else None,
            'like_count': statistics['likeCount'] if 'likeCount' in statistics else None,
            'comment_count': statistics['commentCount'] if 'commentCount' in statistics else None
        }
        videos.append(video_info)

    # Save the videos to a CSV file
    csv_file = f'/home/tianyu/YouTube_Data_Science/time_series/videos/{date_str}.csv'
    # csv_file = f'{HISTORY_VIDEO_DIR}/{date_str}.csv'
    # Create the videos
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=videos[0].keys())
            writer.writeheader()

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=videos[0].keys())
        for video in videos:
            writer.writerow(video)
        # Print the number of videos saved
        print(f'{len(videos)} videos saved to {csv_file} from {date_str}.')

    return csv_file


def main():
    for channel_name, channel_id in CHANNEL_IDS.items():
        csv_file = get_channel_videos_14_days_ago(channel_id)
        print(f'Videos from {channel_name} saved to {csv_file}.')


if __name__ == '__main__':
    main()
