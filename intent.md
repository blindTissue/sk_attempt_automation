# Goal: Identify transitions in a DJ mix recording and create a video clip for each one, including the full transition plus 10 seconds before and after it.

## Details

Identify transitions between tracks by recognizing each track’s main melody. For each transition:
- Extract the audio from 10 seconds before the transition starts to 10 seconds after it ends, keeping the total clip length at no more than 60 seconds.
- Find the album covers for both tracks and place them side by side with a multiplication symbol (×) in the middle. Include each track’s title and artist name beneath its cover.
- Combine this image with the extracted audio to create an Instagram-ready video.
Since the recording contains multiple tracks, generate a separate video for each transition.

## I/O
Input: wav or mp3
Output: videos

## Music Recognition

I have Shazam-API that could be used for music recognition. However, each call cost money. Try to be as frugal as possible.